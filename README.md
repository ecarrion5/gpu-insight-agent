# GPU Insight Agent — a hands-on agentic data-analysis prototype

A small, real agentic pipeline: it profiles messy multi-year GPU performance data with
deterministic code, then runs an LLM-driven **plan > generate pandas > execute in a
sandbox > ground the insight** loop to surface non-obvious findings. Built to mirror a problem 
a GPU team would solve, and to demonstrate hands-on engineering.

## What it does
1. Generates 5 "years" of synthetic GPU data with **schema drift** (each year names the
   same metric differently), a **null spike**, and an **embedded hidden pattern**.
2. **Normalizes** it deterministically (code, not an LLM).
3. **Profiles** it and hands the model a compact schema — never the raw data.
4. Runs an agent loop that writes pandas, **executes it in a sandboxed process**, and
   turns real results into **Pydantic-validated, grounded insights**.

## Run it

```bash
pip install -r requirements.txt

# Option A — OpenAI
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini
python main.py

# Option B — fully local, no API key (Ollama)
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export LLM_MODEL=llama3.1
python main.py
```

The deterministic parts (data, profiling, sandbox) run with **no API key** — try:
```bash
python make_sample_data.py      # writes + previews the data
```

## Files — and what each one does

| File | What it is | What it does |
|---|---|---|
| `make_sample_data.py` | Generates + **normalizes** messy data | "I don't point an LLM at raw data. I wrote the normalization that reconciles schema drift across 5 years first." |
| `profiler.py` | Deterministic profiling > compact schema | "I profile with pandas and send the model a schema summary, not the frame - that's the token control lever. Wide frames blow up token cost otherwise." |
| `sandbox.py` | AST validation + isolated process + timeout | "Shouldn't `exec()` model output in-process. AST-reject imports/dunders, run in a separate process, hard-kill on timeout, restrict the namespace." |
| `schemas.py` | Pydantic output contract | "Structured output enforced by Pydantic; a malformed response raises `ValidationError` and the loop re-prompts. Same pattern I used in Thematica between stages." |
| `agent.py` | The plan > execute > ground loop | "I log every genrated query with a hash and spot-check it, because the #1 failure is valid pandas answering the wrong question. Insights are grounded in the *executed result*, never the model's imagination." |
| `llm_client.py` | Single model-access module | "One place for model + credentials. Swapping OpenAI for local vLLM/Ollama or a gateway is a one-line `base_url` change." |
| `main.py` | Entrypoint | "The `__main__` guard is required — the sandbox spawns processes and would recursively re-import without it." |

## Failure modes I built for
- **Runaway token cost** - compact schema instead of raw data; targeted context.
- **Valid-but-wrong pandas** - query hashing + verbose logging + spot-checks.
- **Unsafe code execution** - AST allow-listing + isolated process + timeout + restricted builtins.
- **Malformed model output** - Pydantic validation + retry-with-error re-prompt.
- **Hallucinated findings** - insights grounded strictly in executed results.
- **One agent failing the run** - failed branches are skipped; the loop replans.

## Limitations!
- The sandbox is **prototype-grade**, not a hardened boundary for adversarial code.
  For Production: container with no network, read-only FS, etc, or a hosted code-interpreter sandbox.
- I pickle the whole frame to the worker; for real 5-year data I'd pass a **path** or
  use shared memory / a warehouse, and push compute down to the data.
- `Manager().Queue()` per call has overhead; production would use a **persistent sandbox
  service**, not a fresh process per query.
- No caching yet; I'd add **response caching keyed by prompt+code hash** to cut cost.

## One level up (how I'd productionize)
Warehouse the normalized data with lineage (dbt/OpenLineage) · vector store over metric
metadata + past analyses as agent memory · LangGraph for an explicit, checkpointed state
machine with a human-approval node · eval harness with gold findings run in CI ·
run the inference on Instinct GPUs (the case-study angle: tokens/dollar as a platform proof).
