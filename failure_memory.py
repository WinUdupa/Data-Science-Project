"""
failure_memory.py

Tracks failed and low-scoring feature functions across iterations and injects
a "do not repeat these mistakes" section into every LLM prompt.

How it works:
    - Every evaluated feature is recorded with its score and error (if any)
    - Features that errored or scored below a threshold are stored as failures
    - Before each API call, a failure summary is appended to the prompt
    - The LLM sees what didn't work and avoids repeating those patterns

Usage:
    memory = FailureMemory(score_threshold=0.0)
    memory.record(feature_fn="...", score=None, error="math domain error")
    memory.record(feature_fn="...", score=0.79, error=None)

    # Inject into prompt before API call:
    enriched_prompt = memory.inject(base_prompt)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class FailureRecord:
    feature_fn:   str
    error:        Optional[str]   # exception message if it crashed
    score:        Optional[float] # None = crashed, float = low score
    iteration:    int
    failure_type: str             # "crash" | "low_score"


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class FailureMemory:
    """
    Maintains a rolling memory of failed feature attempts and injects
    failure summaries into LLM prompts to prevent repeated mistakes.
    """

    FAILURE_SECTION_TEMPLATE = """\

--- FAILURE MEMORY (do NOT repeat these patterns) ---
The following approaches have already been tried and FAILED or scored poorly.
Avoid these exact patterns and the underlying mistakes they represent:

{failure_summary}
--- END FAILURE MEMORY ---
"""

    CRASH_ENTRY_TEMPLATE = """\
[CRASH - Iteration {iteration}] Error: "{error}"
Problematic pattern:
{snippet}
"""

    LOW_SCORE_ENTRY_TEMPLATE = """\
[LOW SCORE {score:.4f} - Iteration {iteration}]
Problematic pattern:
{snippet}
"""

    def __init__(
        self,
        score_threshold: float = 0.0,   # scores below this are treated as failures
        max_failures:    int   = 10,     # max failures to show in prompt (avoid bloat)
        snippet_lines:   int   = 6,      # how many lines of bad code to show
        log_path:        Optional[str] = None,
    ):
        self.score_threshold = score_threshold
        self.max_failures    = max_failures
        self.snippet_lines   = snippet_lines
        self.log_path        = log_path

        self._records: list[FailureRecord] = []
        self._iteration = 0

    # ── Public API ──────────────────────────────────────────────

    def record(
        self,
        feature_fn: str,
        score:      Optional[float],
        error:      Optional[str] = None,
    ):
        """
        Record a feature attempt. Call this after every evaluation.

        Args:
            feature_fn: The generated Python function string.
            score:      Evaluation score. None = crashed during execution.
            error:      Exception message if execution failed.
        """
        is_crash     = score is None or error is not None
        is_low_score = (not is_crash) and score is not None and score < self.score_threshold

        if is_crash or is_low_score:
            failure_type = "crash" if is_crash else "low_score"
            self._records.append(FailureRecord(
                feature_fn   = feature_fn,
                error        = error,
                score        = score,
                iteration    = self._iteration,
                failure_type = failure_type,
            ))
            self._save_log()

        self._iteration += 1

    def inject(self, prompt: str) -> str:
        """
        Append a failure summary to the prompt.
        Returns the original prompt unchanged if there are no failures.
        """
        if not self._records:
            return prompt

        summary = self._build_summary()
        return prompt + self.FAILURE_SECTION_TEMPLATE.format(failure_summary=summary)

    def count(self) -> int:
        return len(self._records)

    def get_records(self) -> list[FailureRecord]:
        return list(self._records)

    def clear(self):
        """Reset memory — useful between dataset splits."""
        self._records.clear()
        self._iteration = 0

    # ── Internal ────────────────────────────────────────────────

    def _snippet(self, feature_fn: str) -> str:
        """Return a short snippet of the function to show the LLM."""
        lines = feature_fn.strip().splitlines()
        # Show first N non-empty lines
        visible = [l for l in lines if l.strip()][:self.snippet_lines]
        return "\n".join(visible)

    def _build_summary(self) -> str:
        """Build a readable failure summary for injection into the prompt."""
        # Show most recent failures first, capped at max_failures
        recent = self._records[-self.max_failures:]
        parts  = []

        # Group by type for clarity
        crashes    = [r for r in recent if r.failure_type == "crash"]
        low_scores = [r for r in recent if r.failure_type == "low_score"]

        # Deduplicate by error message to avoid repeating the same mistake
        seen_errors = set()
        for r in crashes:
            key = r.error or "unknown"
            if key in seen_errors:
                continue
            seen_errors.add(key)
            parts.append(self.CRASH_ENTRY_TEMPLATE.format(
                iteration = r.iteration,
                error     = r.error or "unknown error",
                snippet   = self._snippet(r.feature_fn),
            ))

        # Deduplicate low scores by score bucket
        for r in low_scores:
            parts.append(self.LOW_SCORE_ENTRY_TEMPLATE.format(
                score     = r.score or 0.0,
                iteration = r.iteration,
                snippet   = self._snippet(r.feature_fn),
            ))

        return "\n".join(parts) if parts else "None yet."

    def _save_log(self):
        if not self.log_path:
            return
        os.makedirs(self.log_path, exist_ok=True)
        out = os.path.join(self.log_path, "failure_memory.json")
        records = [
            {
                "iteration":    r.iteration,
                "failure_type": r.failure_type,
                "error":        r.error,
                "score":        r.score,
                "feature_fn":   r.feature_fn[:500],  # truncate for storage
            }
            for r in self._records
        ]
        with open(out, "w") as f:
            json.dump(records, f, indent=2)