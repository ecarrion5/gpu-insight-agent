"""Deterministic data profiling. No LLM here, on purpose.

Main points:
Don't point an LLM at raw, messy data. Characterize the data 
with plain pandas first, then hand the agent a compact SCHEMA SUMMARY instead 
of the dataframe itself. Pointing an LLM at dirty, unschematized data just 
burns tokens producing confident nonsense — and a wide frame sends every column
name and dtype on every single call.
"""

import pandas as pd


def profile(df: pd.DataFrame) -> dict:
    """A structured, deterministic description of the dataframe."""
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [
            {
                "name": c,
                "dtype": str(df[c].dtype),
                "null_pct": round(float(df[c].isna().mean()) * 100, 1),
                "n_unique": int(df[c].nunique(dropna=True)),
                "sample": [str(v) for v in df[c].dropna().head(3).tolist()],
            }
            for c in df.columns
        ],
    }


def compact_schema(p: dict) -> str:
    """Token-frugal schema string for the LLM. This is the 'input projection' lever:
    we send a summary, never the data."""
    lines = [f"{p['n_cols']} columns, {p['n_rows']:,} rows"]
    for col in p["columns"]:
        lines.append(
            f"- {col['name']} ({col['dtype']}, {col['null_pct']}% null, "
            f"{col['n_unique']} unique) e.g. {col['sample']}"
        )
    return "\n".join(lines)
