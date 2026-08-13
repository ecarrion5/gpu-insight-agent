"""The analysis agent: an explicit plan -> execute -> ground loop.

Main points:
  * RETRY ON VALIDATION ERROR: the planner's output is validated by Pydantic; on
    failure we re-prompt WITH the error message. (Thematica technique.)
  * QUERY LOGGING: every generated query is hashed + logged (verbose). This is the
    mitigation for the #1 failure mode of data agents -- writing valid pandas that
    answers the WRONG question. You log it and spot-check it.
  * GROUNDING: an Insight is only produced from the ACTUAL executed result, never from
    the model's imagination. That's hallucination control for data.
  * TEMPERATURE AS A KNOB: planning uses higher temp (we want diverse questions);
    grounding/summarizing uses temp 0 (we want reproducibility).
  * DETERMINISTIC PROFILING: the model sees a compact schema, not the dataframe.
"""

import json
import hashlib
import pandas as pd

# import supporting modulde
from schemas import AnalysisStep, Insight
from llm_client import LLMClient
from sandbox import run_pandas
from profiler import profile, compact_schema

# define the system prompts for planning and insight generation
PLAN_SYS = (
    "You are a data analyst exploring GPU performance data for NON-OBVIOUS insights.\n"
    "Given a schema, propose ONE specific analytical question and the pandas code to\n"
    "answer it. Rules: the code must assign its answer to a variable named `result`;\n"
    "use only pandas (as pd), numpy (as np), and the dataframe `df`; no imports.\n"
    'Return ONLY JSON: {"question": "...", "pandas_code": "..."}'
)

INSIGHT_SYS = (
    "You summarize an executed analysis into a grounded insight. Base every claim on\n"
    "the provided RESULT only. Return ONLY JSON with keys: finding, supporting_question,\n"
    "supporting_code, result_summary, confidence (low|medium|high), "
    "novelty (expected|surprising)."
)


def _json_from(text: str) -> dict:
    """Tolerant JSON extraction: models sometimes wrap JSON in prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    # strict=False: local models (e.g. Ollama/llama3.1) often emit literal newlines
    # inside string values instead of escaping them as \n; OpenAI's models don't.
    return json.loads(text[start : end + 1], strict=False)


def _plan_step(llm: LLMClient, schema: str, asked: list[str], max_retries: int = 3) -> AnalysisStep:
    base = (
        f"SCHEMA:\n{schema}\n\n"
        f"Questions already asked (do NOT repeat):\n{asked}\n\n"
        "Propose the next question and its pandas code."
    )
    last_err = ""
    for _ in range(max_retries):
        prompt = base + (f"\n\nYour previous attempt was rejected: {last_err}" if last_err else "")
        raw = llm.complete(PLAN_SYS, prompt, temperature=0.7, json_mode=True)  # exploration -> diversity
        try:
            return AnalysisStep(**_json_from(raw))             # Pydantic validates; may raise
        except Exception as e:                                 # ValidationError or bad JSON
            last_err = str(e)
    raise RuntimeError(f"planning failed after {max_retries} attempts: {last_err}")


def analyze(df: pd.DataFrame, n_insights: int = 5, verbose: bool = True) -> list[Insight]:
    llm = LLMClient()
    schema = compact_schema(profile(df))     # deterministic + token-frugal
    insights: list[Insight] = []
    asked: list[str] = []
    guard = 0

    while len(insights) < n_insights and guard < n_insights * 4:
        guard += 1  # keep the loop from running forever if the model keeps failing
        step = _plan_step(llm, schema, asked)
        asked.append(step.question)
        code_hash = hashlib.md5(step.pandas_code.encode()).hexdigest()[:8]

        status, payload = run_pandas(step.pandas_code, df)   # sandboxed + hard timeout

        if verbose:
            print(f"[{code_hash}] Q: {step.question}")
            print(f"          code: {' '.join(step.pandas_code.split())[:110]}")
            print(f"          -> {status}: {str(payload)[:150]}")

        if status != "ok":
            continue  # skip the failed branch; the loop will plan a different question

        # Otherwise, we have a good question. Continue to get the insight.
        # Ground the insight in the REAL result, not the model's imagination.
        user = (
            f"QUESTION: {step.question}\nCODE: {step.pandas_code}\n"
            f"EXECUTED RESULT: {payload}\nSummarize as JSON."
        )
        raw = llm.complete(INSIGHT_SYS, user, temperature=0.0, json_mode=True)  # grounding -> deterministic
        try:
            insights.append(Insight(**_json_from(raw)))
        except Exception as e:
            if verbose:
                print(f"          insight rejected: {e}")

    return insights
