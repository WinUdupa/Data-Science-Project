"""
prompt_evolution.py

Handles meta-optimization of the instruction prompt used by the LLM sampler.

How it works:
    Every `evolution_interval` iterations, the top-scoring feature functions
    collected so far are sent back to the LLM with a meta-prompt asking it to
    reflect on what worked and rewrite the instruction prompt to be better.
    The evolved prompt replaces the current one for all subsequent iterations.

Usage:
    evolve = PromptEvolver(api_model="llama-3.3-70b-versatile", evolution_interval=5)
    evolve.record(score=0.83, feature_fn="def modify_features...")
    new_prompt = evolve.maybe_evolve(current_prompt, iteration=5)
"""

from __future__ import annotations

import os
import json
import http.client
import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class FeatureRecord:
    """Stores a generated feature function and its evaluation score."""
    score:      float
    feature_fn: str
    iteration:  int
    prompt_version: int


@dataclass
class PromptEvolutionState:
    """Full history of prompts and their performance."""
    history: list[dict] = field(default_factory=list)

    def add(self, version: int, prompt: str, avg_score: float):
        self.history.append({
            "version":   version,
            "prompt":    prompt,
            "avg_score": avg_score,
        })

    def best_prompt(self) -> str:
        """Return the prompt that produced the highest average score."""
        if not self.history:
            return ""
        return max(self.history, key=lambda x: x["avg_score"])["prompt"]

    def to_json(self) -> str:
        return json.dumps(self.history, indent=2)


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class PromptEvolver:
    """
    Manages prompt evolution by periodically asking the LLM to improve
    its own instruction prompt based on which features scored highest.
    """

    BASE_PROMPT = (
        "You are a helpful assistant tasked with discovering new features / "
        "dropping less important features for the given prediction task. "
        "Complete the 'modify_features' function below, considering the "
        "physical meaning and relationships of inputs."
    )

    META_PROMPT_TEMPLATE = """\
You are a prompt engineer for an automated feature engineering system.

Your job is to improve the instruction prompt that is given to an LLM to generate
new features for a machine learning dataset. The current prompt is:

--- CURRENT PROMPT ---
{current_prompt}
--- END CURRENT PROMPT ---

Here are the TOP PERFORMING feature functions discovered so far (highest scores first):

{top_features}

Here are FAILED or LOW-SCORING approaches to avoid:

{bad_features}

Based on this evidence:
1. What patterns made the top features successful?
2. What should the LLM focus on or avoid?
3. Write an IMPROVED instruction prompt that will guide the LLM to generate
   even better features in the next round of iterations.

Return ONLY the new instruction prompt text. No explanations, no preamble.
"""

    def __init__(
        self,
        api_model:          str   = "llama-3.3-70b-versatile",
        evolution_interval: int   = 5,
        top_k:              int   = 3,
        bad_k:              int   = 2,
        max_retries:        int   = 5,
        log_path:           Optional[str] = None,
    ):
        """
        Args:
            api_model:          Groq/OpenAI model name.
            evolution_interval: Evolve the prompt every N iterations.
            top_k:              Number of top features to show the LLM.
            bad_k:              Number of bad features to show as negative examples.
            max_retries:        API retry limit for the meta-call.
            log_path:           If set, saves prompt history as JSON here.
        """
        self.api_model          = api_model
        self.evolution_interval = evolution_interval
        self.top_k              = top_k
        self.bad_k              = bad_k
        self.max_retries        = max_retries
        self.log_path           = log_path

        self.current_prompt     = self.BASE_PROMPT
        self.prompt_version     = 0
        self.iteration          = 0

        self._records: list[FeatureRecord] = []
        self.state = PromptEvolutionState()
        self.state.add(version=0, prompt=self.current_prompt, avg_score=0.0)

    # ── Public API ──────────────────────────────────────────────

    def record(self, score: Optional[float], feature_fn: str):
        """
        Call this after every evaluated feature function.
        score=None means the feature errored — treated as very low score.
        """
        self._records.append(FeatureRecord(
            score        = score if score is not None else -999.0,
            feature_fn   = feature_fn,
            iteration    = self.iteration,
            prompt_version = self.prompt_version,
        ))
        self.iteration += 1

    def maybe_evolve(self) -> str:
        """
        Check if it's time to evolve. If yes, call the LLM and update the prompt.
        Returns the current (possibly updated) prompt.
        """
        if (self.iteration > 0 and
                self.iteration % self.evolution_interval == 0 and
                len(self._records) >= self.top_k):
            print(f"\n[PromptEvolver] Iteration {self.iteration} — evolving prompt (v{self.prompt_version} → v{self.prompt_version + 1})...")
            evolved = self._call_llm_for_evolution()
            if evolved:
                self.prompt_version += 1
                self.current_prompt  = evolved
                avg = self._avg_recent_score()
                self.state.add(
                    version   = self.prompt_version,
                    prompt    = self.current_prompt,
                    avg_score = avg,
                )
                print(f"[PromptEvolver] Prompt evolved successfully (avg score this round: {avg:.4f})")
                self._save_log()
            else:
                print("[PromptEvolver] Evolution failed — keeping current prompt.")

        return self.current_prompt

    def get_prompt(self) -> str:
        """Return the current instruction prompt."""
        return self.current_prompt

    def get_history(self) -> list[dict]:
        """Return full prompt version history."""
        return self.state.history

    # ── Internal ────────────────────────────────────────────────

    def _top_features(self) -> list[FeatureRecord]:
        valid = [r for r in self._records if r.score > -999.0]
        return sorted(valid, key=lambda r: r.score, reverse=True)[:self.top_k]

    def _bad_features(self) -> list[FeatureRecord]:
        return sorted(self._records, key=lambda r: r.score)[:self.bad_k]

    def _avg_recent_score(self) -> float:
        recent = [r for r in self._records
                  if r.iteration >= self.iteration - self.evolution_interval
                  and r.score > -999.0]
        if not recent:
            return 0.0
        return sum(r.score for r in recent) / len(recent)

    def _format_features(self, records: list[FeatureRecord]) -> str:
        if not records:
            return "None available yet."
        parts = []
        for i, r in enumerate(records, 1):
            parts.append(
                f"[{i}] Score: {r.score:.4f} | Iteration: {r.iteration}\n"
                f"{r.feature_fn[:600]}{'...' if len(r.feature_fn) > 600 else ''}"
            )
        return "\n\n".join(parts)

    def _call_llm_for_evolution(self) -> Optional[str]:
        """Send meta-prompt to LLM and return the evolved instruction prompt."""
        api_key = os.environ.get("API_KEY", "")
        if not api_key:
            print("[PromptEvolver] No API_KEY found — skipping evolution.")
            return None

        meta_prompt = self.META_PROMPT_TEMPLATE.format(
            current_prompt = self.current_prompt,
            top_features   = self._format_features(self._top_features()),
            bad_features   = self._format_features(self._bad_features()),
        )

        # Detect endpoint from model name
        if "gpt" in self.api_model.lower():
            host     = "api.openai.com"
            endpoint = "/v1/chat/completions"
        else:
            host     = "api.groq.com"
            endpoint = "/openai/v1/chat/completions"

        for attempt in range(self.max_retries):
            try:
                conn    = http.client.HTTPSConnection(host)
                payload = json.dumps({
                    "model":      self.api_model,
                    "max_tokens": 512,
                    "messages":   [{"role": "user", "content": meta_prompt}],
                })
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                    "User-Agent":    "LLMFE-PromptEvolver/1.0",
                }
                conn.request("POST", endpoint, payload, headers)
                res  = conn.getresponse()
                data = json.loads(res.read().decode("utf-8"))

                if "error" in data:
                    err = data["error"]
                    if err.get("code") == "rate_limit_exceeded":
                        match = re.search(r"try again in ([0-9.]+)s", err.get("message", ""))
                        wait  = float(match.group(1)) + 1 if match else 60
                        print(f"[PromptEvolver] Rate limited — waiting {wait:.1f}s...")
                        time.sleep(wait)
                    else:
                        print(f"[PromptEvolver] API error: {err}")
                        time.sleep(2)
                    continue

                evolved = data["choices"][0]["message"]["content"].strip()

                # Sanity check — must be non-empty and reasonably long
                if len(evolved) > 30:
                    return evolved

            except Exception as e:
                print(f"[PromptEvolver] Attempt {attempt+1}/{self.max_retries} failed: {e}")
                time.sleep(2)

        return None

    def _save_log(self):
        if not self.log_path:
            return
        os.makedirs(self.log_path, exist_ok=True)
        out = os.path.join(self.log_path, "prompt_evolution_log.json")
        with open(out, "w") as f:
            f.write(self.state.to_json())
        print(f"[PromptEvolver] Log saved to {out}")