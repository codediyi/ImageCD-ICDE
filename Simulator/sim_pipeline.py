#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageCD-ICDE Simulator: DuckDB-based end-to-end pipeline

This script simulates the full pipeline described in the paper-style section:
- LogView (raw interaction logs)
- Data2Imag: aggregate to RawImageView (concept x difficulty per user)
- Imag2Diag: infer dense cognitive image (simulated by noise and simple smoothing/monotonicity)
- Imag2Data: materialize to ProficiencyImageView (a physical table: proficiency_image)
- Query demos: point, concept-level, difficulty-level, range aggregation

Requirements: duckdb, numpy, pandas
"""
from __future__ import annotations
import argparse
import math
import os
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless save
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _banner(title: str) -> None:
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")


def _hr(msg: str) -> None:
    print(f"\n--- {msg} ---")


def generate_synthetic_logs(
    n_users: int,
    n_concepts: int,
    n_difficulties: int,
    items_per_cd: int,
    sparsity: float,
    attempts_low: int,
    attempts_high: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic interaction logs (LogView).

    Columns: user_id, item_id, concept_id, difficulty_id, correctness, timestamp

    - sparsity: expected fraction of (user, concept, difficulty) cells that are empty (0..1)
    - attempts_low/high: inclusive range of number of attempts when a cell is non-empty
    """
    rng = np.random.default_rng(seed)

    # Pre-construct item id pools for each (concept, difficulty)
    # item_id space: 0 .. (n_concepts * n_difficulties * items_per_cd - 1)
    def item_pool(c: int, d: int) -> np.ndarray:
        base = (c * n_difficulties + d) * items_per_cd
        return np.arange(base, base + items_per_cd, dtype=np.int64)

    # Difficulty scale: easier at lower index, harder at higher index
    # Map difficulty_id in [0, n_difficulties-1] to a numeric level in [-1.5, 1.5]
    diff_levels = np.linspace(-1.5, 1.5, n_difficulties)

    rows = []
    start_time = datetime(2024, 1, 1, 8, 0, 0)

    for u in range(n_users):
        # Latent traits per user
        ability = rng.normal(loc=0.0, scale=1.0)  # global ability
        concept_affinity = rng.normal(loc=0.0, scale=0.7, size=n_concepts)  # user-specific concept affinity
        noise_scale = 0.3
        diff_weight = 1.1  # how strongly difficulty depresses success probability

        for c in range(n_concepts):
            for d in range(n_difficulties):
                # Sparsity: skip many cells to yield a sparse RawImage
                if rng.random() < sparsity:
                    continue

                # Determine attempts
                n_attempts = int(rng.integers(low=attempts_low, high=attempts_high + 1))
                if n_attempts <= 0:
                    continue

                # Success probability for this (u, c, d)
                base_logit = ability + concept_affinity[c] - diff_weight * diff_levels[d]
                p = float(sigmoid(base_logit + rng.normal(0, noise_scale)))
                p = min(max(p, 0.02), 0.98)

                # Sample attempts
                pool = item_pool(c, d)
                for k in range(n_attempts):
                    item_id = int(rng.choice(pool))
                    correct = int(rng.random() < p)
                    # Simulate timestamps with small offsets
                    ts = start_time + timedelta(minutes=int(rng.integers(0, 60 * 24 * 60)))
                    rows.append((u, item_id, c, d, correct, ts))

    df = pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "item_id",
            "concept_id",
            "difficulty_id",
            "correctness",
            "timestamp",
        ],
    )
    return df


def create_duckdb(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    return con


def load_logview(con: duckdb.DuckDBPyConnection, df_logs: pd.DataFrame) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE log_view AS
        SELECT * FROM df_logs
        """
    )
    # Helpful sort index
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_log_user_concept_difficulty ON log_view(user_id, concept_id, difficulty_id)"
    )
    cnt = con.execute("SELECT COUNT(*) FROM log_view").fetchone()[0]
    print(f"LogView ready: {cnt:,} rows")


def aggregate_raw_image(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE raw_image_view AS
        SELECT
            user_id,
            concept_id,
            difficulty_id,
            COUNT(*) AS attempts,
            SUM(CASE WHEN correctness = 1 THEN 1 ELSE 0 END) AS corrects,
            CASE WHEN COUNT(*) = 0 THEN 0.0 ELSE SUM(CASE WHEN correctness = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) END AS accuracy
        FROM log_view
        GROUP BY user_id, concept_id, difficulty_id
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_raw_user_concept_difficulty ON raw_image_view(user_id, concept_id, difficulty_id)"
    )
    cnt = con.execute("SELECT COUNT(*) FROM raw_image_view").fetchone()[0]
    print(f"RawImageView ready: {cnt:,} (user, concept, difficulty) cells")


def imag2diag_simulation(
    con: duckdb.DuckDBPyConnection,
    n_users: int,
    n_concepts: int,
    n_difficulties: int,
    noise_std: float = 0.08,
    seed: int = 123,
) -> None:
    """
    Simulate Imag2Diag by:
    - Pivoting raw_image_view to sparse matrices A_s (accuracy) per user
    - Imputing missing cells with row/col means then global mean
    - Adding Gaussian noise and clamping to [0,1]
    - Enforcing monotonic non-increasing mastery along difficulty (harder shouldn't have higher mastery)
    - Writing results to proficiency_image table
    """
    rng = np.random.default_rng(seed)

    _hr("Imag2Diag: create output table and clear previous data")
    # Ensure output table exists empty
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS proficiency_image (
            user_id INTEGER,
            concept_id INTEGER,
            difficulty_id INTEGER,
            mastery_value DOUBLE
        )
        """
    )
    con.execute("DELETE FROM proficiency_image")

    # Load raw_image into pandas for processing per user
    _hr("Load RawImageView into memory (pandas)")
    df_raw = con.execute("SELECT * FROM raw_image_view").fetch_df()

    # Precompute global mean accuracy for fallback
    global_mean = 0.5
    if not df_raw.empty:
        global_mean = float(df_raw["accuracy"].mean())
    print(f"Global mean accuracy (fallback): {global_mean:.4f}")

    # For speed, build a dict per user
    users = sorted(df_raw["user_id"].unique().tolist()) if not df_raw.empty else list(range(n_users))
    print(f"Users to process: {len(users):,}")
    insert_rows = []

    for u in users:
        df_u = df_raw[df_raw["user_id"] == u]
        # Build sparse grid A_s
        A = np.full((n_concepts, n_difficulties), np.nan, dtype=float)
        if not df_u.empty:
            for _, r in df_u.iterrows():
                ci = int(r["concept_id"])
                di = int(r["difficulty_id"])
                A[ci, di] = float(r["accuracy"])

        # Impute missing values: row mean -> column mean -> global mean
        # Row means
        row_means = np.nanmean(A, axis=1)
        inds = np.where(np.isnan(A))
        if np.any(np.isnan(row_means)):
            # Replace any nan row means with global
            row_means = np.where(np.isnan(row_means), global_mean, row_means)
        # Fill by row means first
        for (ci, di) in zip(*inds):
            if np.isnan(A[ci, di]):
                A[ci, di] = row_means[ci]
        # Still possible to have NaN if entire row was NaN originally; fill by column means
        col_means = np.nanmean(A, axis=0)
        col_means = np.where(np.isnan(col_means), global_mean, col_means)
        inds2 = np.where(np.isnan(A))
        for (ci, di) in zip(*inds2):
            A[ci, di] = col_means[di]
        # Global fill as last resort
        A = np.where(np.isnan(A), global_mean, A)

        # Add Gaussian noise and clamp
        M = A + rng.normal(loc=0.0, scale=noise_std, size=A.shape)
        M = np.clip(M, 0.0, 1.0)

        # Enforce monotonic non-increasing along difficulty dimension
        # For each concept row, ensure M[c, d] <= M[c, d-1]
        for c in range(n_concepts):
            for d in range(1, n_difficulties):
                if M[c, d] > M[c, d - 1]:
                    M[c, d] = M[c, d - 1]

        # Collect rows
        for c in range(n_concepts):
            for d in range(n_difficulties):
                insert_rows.append((int(u), int(c), int(d), float(M[c, d])))

    _hr("Materialize proficiency_image table (Imag2Data)")
    # Insert into DuckDB
    df_M = pd.DataFrame(insert_rows, columns=["user_id", "concept_id", "difficulty_id", "mastery_value"])
    con.execute("INSERT INTO proficiency_image SELECT * FROM df_M")

    # Indexes for query performance
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_prof_user_concept_difficulty ON proficiency_image(user_id, concept_id, difficulty_id)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_prof_user_difficulty_concept ON proficiency_image(user_id, difficulty_id, concept_id)"
    )
    cnt = con.execute("SELECT COUNT(*) FROM proficiency_image").fetchone()[0]
    print(f"ProficiencyImageView ready: {cnt:,} rows (materialized)")


def demo_queries(con: duckdb.DuckDBPyConnection, user_id: int, concept_id: int, difficulty_id: int) -> None:
    print("\n--- Demo Queries ---")

    # Point query
    q1 = con.execute(
        """
        SELECT * FROM proficiency_image
        WHERE user_id = ? AND concept_id = ? AND difficulty_id = ?
        """,
        [user_id, concept_id, difficulty_id],
    ).fetch_df()
    print("Point Query (user, concept, difficulty):")
    print(q1)

    # Concept-level distribution for a user
    q2 = con.execute(
        """
        SELECT difficulty_id, mastery_value
        FROM proficiency_image
        WHERE user_id = ? AND concept_id = ?
        ORDER BY difficulty_id
        """,
        [user_id, concept_id],
    ).fetch_df()
    print("\nConcept-level (fixed user, concept; vary difficulty):")
    print(q2.head(10))

    # Difficulty-level distribution across concepts for a user
    q3 = con.execute(
        """
        SELECT concept_id, mastery_value
        FROM proficiency_image
        WHERE user_id = ? AND difficulty_id = ?
        ORDER BY concept_id
        """,
        [user_id, difficulty_id],
    ).fetch_df()
    print("\nDifficulty-level (fixed user, difficulty; vary concept):")
    print(q3.head(10))

    # Range heatmap export: subset of concepts and difficulty range
    # Example: concepts in [0..min(4, maxC)] and difficulties in [0..min(2, maxD)]
    c_hi = int(min(4, con.execute("SELECT MAX(concept_id) FROM proficiency_image").fetchone()[0]))
    d_hi = int(min(2, con.execute("SELECT MAX(difficulty_id) FROM proficiency_image").fetchone()[0]))

    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / f"cognitive_image_u{user_id}_c0-{c_hi}_d0-{d_hi}.png"
    _hr("Range heatmap (subset concepts/difficulties): save to image")
    save_heatmap_subset(
        con=con,
        user_id=user_id,
        c_lo=0,
        c_hi=c_hi,
        d_lo=0,
        d_hi=d_hi,
        out_path=out_path,
    )
    print(f"Heatmap saved: {out_path}")


def save_heatmap_subset(
    con: duckdb.DuckDBPyConnection,
    user_id: int,
    c_lo: int,
    c_hi: int,
    d_lo: int,
    d_hi: int,
    out_path: Path,
) -> None:
    """Save a heatmap image for a user over a subrange of (concept, difficulty)."""
    df = con.execute(
        """
        SELECT concept_id, difficulty_id, mastery_value
        FROM proficiency_image
        WHERE user_id = ? AND concept_id BETWEEN ? AND ? AND difficulty_id BETWEEN ? AND ?
        ORDER BY concept_id, difficulty_id
        """,
        [user_id, c_lo, c_hi, d_lo, d_hi],
    ).fetch_df()

    if df.empty:
        raise ValueError("No data returned for the specified range to plot heatmap.")

    nC = c_hi - c_lo + 1
    nD = d_hi - d_lo + 1
    mat = np.full((nC, nD), np.nan, dtype=float)
    for _, r in df.iterrows():
        ci = int(r["concept_id"]) - c_lo
        di = int(r["difficulty_id"]) - d_lo
        mat[ci, di] = float(r["mastery_value"])

    # Robust fill in case of any missing cells (shouldn't happen due to dense materialization)
    if np.isnan(mat).any():
        fill = float(np.nanmean(mat)) if not np.isnan(np.nanmean(mat)) else 0.5
        mat = np.where(np.isnan(mat), fill, mat)

    plt.figure(figsize=(max(6, nD * 0.8), max(4, nC * 0.5)))
    im = plt.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto", origin="upper")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="mastery")
    plt.title(f"User {user_id} Mastery Heatmap (Concept {c_lo}-{c_hi}, Difficulty {d_lo}-{d_hi})")
    plt.xlabel("difficulty_id")
    plt.ylabel("concept_id")
    plt.xticks(ticks=np.arange(0, nD), labels=[str(x) for x in range(d_lo, d_hi + 1)])
    plt.yticks(ticks=np.arange(0, nC), labels=[str(x) for x in range(c_lo, c_hi + 1)])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ImageCD DuckDB Simulator")
    parser.add_argument("--db", type=str, default=None, help="Path to DuckDB file (default: Simulator/simulator.duckdb)")
    parser.add_argument("--users", type=int, default=50, help="Number of users")
    parser.add_argument("--concepts", type=int, default=12, help="Number of concepts")
    parser.add_argument("--difficulties", type=int, default=8, help="Number of difficulty bins")
    parser.add_argument("--items-per-cd", type=int, default=20, help="Items per (concept, difficulty)")
    parser.add_argument("--sparsity", type=float, default=0.6, help="Fraction of empty (user,concept,difficulty) cells [0..1]")
    parser.add_argument("--attempts-low", type=int, default=1, help="Min attempts per non-empty cell")
    parser.add_argument("--attempts-high", type=int, default=5, help="Max attempts per non-empty cell")
    parser.add_argument("--noise-std", type=float, default=0.08, help="Gaussian noise std for Imag2Diag simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Resolve DB path
    script_dir = Path(__file__).resolve().parent
    db_path = Path(args.db) if args.db else script_dir / "simulator.duckdb"

    _banner("ImageCD-ICDE Pipeline Simulator (DuckDB)")
    print("This script simulates the full pipeline described in the paper-style section:")
    print("- LogView (raw interaction logs)")
    print("- Data2Imag: aggregate to RawImageView (concept x difficulty per user)")
    print("- Imag2Diag: infer dense cognitive image (simulated by noise + monotonicity)")
    print("- Imag2Data: materialize to ProficiencyImageView (table: proficiency_image)")
    print("- Query demos: point, concept-level, difficulty-level, range aggregation")

    _hr("Parameters")
    print({
        "users": args.users,
        "concepts": args.concepts,
        "difficulties": args.difficulties,
        "items_per_cd": args.items_per_cd,
        "sparsity": args.sparsity,
        "attempts_range": (args.attempts_low, args.attempts_high),
        "noise_std": args.noise_std,
        "seed": args.seed,
    })

    _hr("Generate synthetic LogView")
    df_logs = generate_synthetic_logs(
        n_users=args.users,
        n_concepts=args.concepts,
        n_difficulties=args.difficulties,
        items_per_cd=args.items_per_cd,
        sparsity=args.sparsity,
        attempts_low=args.attempts_low,
        attempts_high=args.attempts_high,
        seed=args.seed,
    )
    print(f"Logs generated: {len(df_logs):,} rows")

    _hr("Create DuckDB and load LogView")
    print(f"Creating DuckDB at: {db_path}")
    con = create_duckdb(db_path)

    load_logview(con, df_logs)

    _hr("Data2Imag: aggregate RawImageView")
    aggregate_raw_image(con)

    _hr("Imag2Diag + Imag2Data: simulate and materialize ProficiencyImageView")
    imag2diag_simulation(
        con,
        n_users=args.users,
        n_concepts=args.concepts,
        n_difficulties=args.difficulties,
        noise_std=args.noise_std,
        seed=args.seed + 1,
    )

    # Basic counts
    log_cnt = con.execute("SELECT COUNT(*) FROM log_view").fetchone()[0]
    raw_cnt = con.execute("SELECT COUNT(*) FROM raw_image_view").fetchone()[0]
    prof_cnt = con.execute("SELECT COUNT(*) FROM proficiency_image").fetchone()[0]
    print(f"\nRow counts: logs={log_cnt:,}, raw_image_view={raw_cnt:,}, proficiency_image={prof_cnt:,}")

    _hr("Demo queries")
    # Demo queries using user 0 (or min user) and first concept/difficulty
    u_min = int(con.execute("SELECT COALESCE(MIN(user_id), 0) FROM proficiency_image").fetchone()[0])
    demo_queries(con, user_id=u_min, concept_id=0, difficulty_id=0)

    _banner("Done")
    print("You can inspect the DuckDB file at:")
    print(str(db_path))


if __name__ == "__main__":
    main()
