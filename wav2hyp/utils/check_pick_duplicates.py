#!/usr/bin/env python3
"""Check picks/eqt-volpick.h5 for duplicate picks (same trace_id, peak_time, phase)."""
import pandas as pd
import sys

def main():
    path = "results/sthelens/picks/eqt-volpick.h5"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    picks = pd.read_hdf(path, key="picks")
    print("Picks table shape:", picks.shape)
    print("Columns:", list(picks.columns))
    print()

    key_cols = ["trace_id", "peak_time", "phase"]
    dup = picks.duplicated(subset=key_cols, keep=False)
    n_dup_rows = int(dup.sum())

    print("--- Exact duplicates (same trace_id, peak_time, phase) ---")
    print("Rows involved in duplicates:", n_dup_rows)
    if n_dup_rows > 0:
        dup_df = picks.loc[dup].sort_values(key_cols)
        n_groups = dup_df.groupby(key_cols).ngroups
        print("Unique (trace_id, peak_time, phase) with >1 row:", n_groups)
        print("\nSample of duplicate rows (first 24):")
        print(dup_df.head(24).to_string())
    else:
        print("No exact duplicates found.")

    full_dup = picks.duplicated(keep=False)
    print("\n--- Full row duplicates ---")
    print("Rows that are exact copies of another row:", int(full_dup.sum()))

if __name__ == "__main__":
    main()
