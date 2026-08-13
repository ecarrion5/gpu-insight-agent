"""Demo entrypoint. Run:  python main.py

The __main__ guard is REQUIRED, not optional: the sandbox uses multiprocessing with
the 'spawn' start method, which re-imports this module in the child process. Without
the guard you'd get a recursive spawn.
"""

from make_sample_data import build_dataset
from agent import analyze


def main() -> None:
    df = build_dataset()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols "
          f"({df['source_year'].min()}-{df['source_year'].max()})\n")

    insights = analyze(df, n_insights=5, verbose=True)

    print("\n=== GROUNDED INSIGHTS ===")
    for i, ins in enumerate(insights, 1):
        print(f"\n{i}. [{ins.confidence} confidence / {ins.novelty}] {ins.finding}")
        print(f"   evidence: {ins.result_summary}")


if __name__ == "__main__":
    main()
