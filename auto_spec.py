"""
auto_spec.py

Automatically generates a specification file for any dataset by analyzing
its structure and asking an LLM to write a domain-aware description.

The spec tells the LLM feature engineer:
    - What each column means
    - What the prediction task is
    - What domain knowledge might help
    - What transformations might be useful

How it works:
    1. Load the CSV and compute basic stats (dtypes, ranges, correlations)
    2. Send a meta-prompt to the LLM asking it to write a spec
    3. Save the spec to ./specs/specification_{dataset}.txt
    4. Return the spec text

Usage:
    spec = generate_spec(
        dataset_path = "./data/heart.csv",
        problem_name = "heart",
        api_model    = "llama-3.3-70b-versatile",
        save_path    = "./specs/specification_heart.txt",
    )
"""

from __future__ import annotations

import http.client
import json
import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# SPEC GENERATION
# ─────────────────────────────────────────────

META_PROMPT_TEMPLATE = """\
You are a data science expert helping to set up an automated feature engineering pipeline.

Analyze the following dataset and write a specification that will guide an LLM to generate
high-quality new features for a machine learning prediction task.

--- DATASET INFO ---
Dataset name : {problem_name}
Task type    : {task_type}
Target column: {target_col}
Rows         : {n_rows}
Columns      : {n_cols}

Column details:
{column_details}

Top correlations with target:
{correlations}
--- END DATASET INFO ---

Write a specification that includes:
1. A brief description of what this dataset represents (domain context)
2. What each input feature means in plain English
3. Which features are likely most important and why
4. What kinds of feature transformations would make sense (interactions, ratios, logs, etc.)
5. Any domain knowledge that could help create better features
6. What to AVOID (e.g. features that would cause data leakage)

Write it clearly so an LLM can use it as context when writing Python feature engineering code.
Do not include any code — only natural language description.
"""


def _summarize_columns(df: pd.DataFrame, target_col: str) -> str:
    """Build a readable column-by-column summary."""
    lines = []
    for col in df.columns:
        dtype  = str(df[col].dtype)
        nuniq  = df[col].nunique()
        if col == target_col:
            lines.append(f"  {col} [TARGET] — dtype: {dtype}, unique values: {nuniq}")
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            mn  = df[col].min()
            mx  = df[col].max()
            med = df[col].median()
            lines.append(f"  {col} — numeric, range [{mn:.2f}, {mx:.2f}], median {med:.2f}, {nuniq} unique")
        else:
            top = df[col].value_counts().index[:3].tolist()
            lines.append(f"  {col} — categorical, {nuniq} unique, top values: {top}")
    return "\n".join(lines)


def _top_correlations(df: pd.DataFrame, target_col: str, top_n: int = 8) -> str:
    """Return top N features most correlated with target."""
    try:
        num_df = df.select_dtypes(include=[np.number])
        if target_col not in num_df.columns:
            return "  (target is not numeric — correlation not computed)"
        corr = num_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        lines = [f"  {col}: {val:.3f}" for col, val in corr.head(top_n).items()]
        return "\n".join(lines)
    except Exception:
        return "  (could not compute correlations)"


def _call_llm(prompt: str, api_model: str, max_retries: int = 5) -> Optional[str]:
    """Call the LLM API and return the response text."""
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        print("[AutoSpec] No API_KEY found — cannot generate spec.")
        return None

    if "gpt" in api_model.lower():
        host, endpoint = "api.openai.com", "/v1/chat/completions"
    else:
        host, endpoint = "api.groq.com", "/openai/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            conn    = http.client.HTTPSConnection(host)
            payload = json.dumps({
                "model":      api_model,
                "max_tokens": 1024,
                "messages":   [{"role": "user", "content": prompt}],
            })
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
                "User-Agent":    "LLMFE-AutoSpec/1.0",
            }
            conn.request("POST", endpoint, payload, headers)
            res  = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))

            if "error" in data:
                err = data["error"]
                if err.get("code") == "rate_limit_exceeded":
                    match = re.search(r"try again in ([0-9.]+)s", err.get("message", ""))
                    wait  = float(match.group(1)) + 1 if match else 60
                    print(f"[AutoSpec] Rate limited — waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    print(f"[AutoSpec] API error: {err}")
                    time.sleep(2)
                continue

            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"[AutoSpec] Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(2)

    return None


# ─────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────

def generate_spec(
    dataset_path: str,
    problem_name: str,
    api_model:    str  = "llama-3.3-70b-versatile",
    save_path:    Optional[str] = None,
    force:        bool = False,
) -> Optional[str]:
    """
    Generate a specification file for a dataset.

    Args:
        dataset_path: Path to the CSV file.
        problem_name: Short name of the dataset (e.g. "heart").
        api_model:    LLM model to use for generation.
        save_path:    Where to save the spec. Defaults to
                      ./specs/specification_{problem_name}.txt
        force:        If False, skip generation if spec already exists.

    Returns:
        The spec text, or None if generation failed.
    """
    REGRESSION_DATASETS = ['forest-fires', 'housing', 'insurance', 'bike', 'wine', 'crab']

    if save_path is None:
        save_path = f"./specs/specification_{problem_name}.txt"

    # Skip if spec already exists and force=False
    if os.path.exists(save_path) and not force:
        print(f"[AutoSpec] Spec already exists at {save_path} — skipping generation.")
        with open(save_path) as f:
            return f.read()

    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"[AutoSpec] Failed to load dataset: {e}")
        return None

    target_col = df.columns[-1]
    is_reg     = problem_name in REGRESSION_DATASETS
    task_type  = "Regression" if is_reg else "Classification"

    # Build meta prompt
    prompt = META_PROMPT_TEMPLATE.format(
        problem_name   = problem_name,
        task_type      = task_type,
        target_col     = target_col,
        n_rows         = len(df),
        n_cols         = len(df.columns),
        column_details = _summarize_columns(df, target_col),
        correlations   = _top_correlations(df, target_col),
    )

    print(f"[AutoSpec] Generating spec for '{problem_name}' using {api_model}...")
    spec = _call_llm(prompt, api_model)

    if spec:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(spec)
        print(f"[AutoSpec] Spec saved to {save_path}")
    else:
        print("[AutoSpec] Spec generation failed.")

    return spec