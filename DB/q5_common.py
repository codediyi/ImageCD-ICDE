# q5_common.py
import os, sys, time, random
from pathlib import Path
from typing import List, Tuple, Optional
import duckdb
import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None

def human_bytes(num_bytes: int) -> str:
    units = ['B','KB','MB','GB','TB']
    size = float(num_bytes)
    for u in units:
        if size < 1024.0:
            return f"{size:.1f} {u}"
        size /= 1024.0
    return f"{size:.1f} PB"

def percentile(values, ps=(50,95,99)):
    if not values: return [float('nan') for _ in ps]
    arr = np.array(values, dtype=float)
    return list(np.percentile(arr, ps))

def ensure_parquet(parquet_path: Path, num_users: int, num_concepts: int, num_difficulties: int,
                   sparsity: float=1.0, seed: int=42) -> None:
    if parquet_path.exists():
        print(f"[SETUP] Using existing parquet: {parquet_path}")
        return
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GEN] Writing synthetic to {parquet_path} ...")
    import pyarrow as pa, pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    users = np.arange(1, num_users+1, dtype=np.int32)
    concepts = np.arange(1, num_concepts+1, dtype=np.int32)
    diffs = np.arange(1, num_difficulties+1, dtype=np.int16)
    pairs = np.array([(c,d) for c in concepts for d in diffs], dtype=np.int32)
    cells_per_user = max(1, int(round(num_concepts*num_difficulties*sparsity)))
    writer = None
    rows = 0
    t0 = time.perf_counter()
    for u in users:
        sel = pairs if cells_per_user >= len(pairs) else pairs[rng.choice(len(pairs), size=cells_per_user, replace=False)]
        prof = rng.beta(3,2,size=len(sel)).astype(np.float32)
        df = pd.DataFrame({
            "user_id": np.full(len(sel), u, dtype=np.int32),
            "concept_id": sel[:,0].astype(np.int32),
            "difficulty_id": sel[:,1].astype(np.int16),
            "proficiency": prof
        })
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path.as_posix(), table.schema, compression="snappy")
        writer.write_table(table)
        rows += len(df)
    if writer: writer.close()
    print(f"[GEN] Done. Total rows: {rows:,}. Elapsed {time.perf_counter()-t0:.1f}s")

def init_rw_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path.as_posix())
    try:
        con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 1))};")
        con.execute("PRAGMA enable_object_cache=true;")
    except Exception:
        pass
    return con

def load_parquet_as_table(con, parquet_path: Path) -> int:
    con.execute("DROP TABLE IF EXISTS ProficiencyImageView;")
    con.execute(f"CREATE TABLE ProficiencyImageView AS SELECT * FROM read_parquet('{parquet_path.as_posix()}');")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ucd ON ProficiencyImageView(user_id,concept_id,difficulty_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_udc ON ProficiencyImageView(user_id,difficulty_id,concept_id);")
    n = con.execute("SELECT COUNT(*) FROM ProficiencyImageView;").fetchone()[0]
    print(f"[DB] TABLE ProficiencyImageView loaded: {n:,} rows.")
    return n

def sample_resources(sec: float=2.0) -> Tuple[Optional[float], Optional[float]]:
    if psutil is None: return None, None
    end = time.time()+sec
    cpu, mem, c = 0.0, 0.0, 0
    while time.time()<end:
        cpu += psutil.cpu_percent(interval=0.2); mem += psutil.virtual_memory().percent; c+=1
    return (cpu/c if c else None, mem/c if c else None)
