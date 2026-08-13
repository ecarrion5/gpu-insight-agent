"""Generates 5 'years' of synthetic GPU performance data, then normalizes it.

Main points:
The messy-data story is real and you have to account for it. Each year uses
different column names for the same metric (schema drift), one year has a null spike,
and there's an embeded non-obvious pattern for the agent to 'discover' (one GPU model
quietly runs hotter per util-point in later years).

build_dataset() is the deterministic normalization step - the code an LLM should NOT
be doing. This is the "understand the data first, with code" point.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "sample_data"

# Same three metrics, named differently each year -> realistic schema drift.
YEAR_SCHEMAS = {
    2021: {"util": "gpu_util_pct",    "temp": "gpu_temp_c",   "power": "power_w"},
    2022: {"util": "utilization",     "temp": "temperature",  "power": "power_w"},
    2023: {"util": "util_percent",    "temp": "temp_c",       "power": "board_power_watts"},
    2024: {"util": "gpu_util",        "temp": "temp_celsius", "power": "power_draw_w"},
    2025: {"util": "utilization_pct", "temp": "gpu_temp",     "power": "power_w"},
}
MODELS = ["MI250", "MI300X", "MI325X"]


def _make_year(year: int, schema: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = 2000
    model = rng.choice(MODELS, size=n)
    util = np.clip(rng.normal(62, 18, n), 0, 100)
    # Hidden pattern: MI300X gains ~1.5C/util-point of extra heat per year after 2021.
    extra_heat = (year - 2021) * 1.5 * (model == "MI300X")
    temp = 40 + util * 0.35 + extra_heat + rng.normal(0, 3, n)
    power = 120 + util * 2.4 + rng.normal(0, 15, n)
    if year == 2023:  # null spike, to exercise the profiler
        temp[rng.random(n) < 0.15] = np.nan
    return pd.DataFrame({
        "timestamp": pd.date_range(f"{year}-01-01", periods=n, freq="4h"),
        "gpu_model": model,
        schema["util"]: util.round(1),
        schema["temp"]: temp.round(1),
        schema["power"]: power.round(0),
    })


def generate_csvs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    for year, schema in YEAR_SCHEMAS.items():
        _make_year(year, schema, rng).to_csv(DATA_DIR / f"gpu_perf_{year}.csv", index=False)


# Maps every drifted column name back to one canonical name.
CANON = {
    "gpu_util_pct": "util_pct", "utilization": "util_pct", "util_percent": "util_pct",
    "gpu_util": "util_pct", "utilization_pct": "util_pct",
    "gpu_temp_c": "temp_c", "temperature": "temp_c", "temp_c": "temp_c",
    "temp_celsius": "temp_c", "gpu_temp": "temp_c",
    "power_w": "power_w", "board_power_watts": "power_w", "power_draw_w": "power_w",
}


def build_dataset() -> pd.DataFrame:
    """Load all yearly CSVs and reconcile their schemas into one clean frame."""
    if not any(DATA_DIR.glob("gpu_perf_*.csv")):
        generate_csvs()
    frames = []
    for csv in sorted(DATA_DIR.glob("gpu_perf_*.csv")):
        d = pd.read_csv(csv, parse_dates=["timestamp"])
        d = d.rename(columns=CANON)                  # deterministic normalization
        d["source_year"] = int(csv.stem.split("_")[-1])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df[["timestamp", "source_year", "gpu_model", "util_pct", "temp_c", "power_w"]]


if __name__ == "__main__":
    generate_csvs()
    print(f"Wrote yearly CSVs to {DATA_DIR}")
    print(build_dataset().head())
