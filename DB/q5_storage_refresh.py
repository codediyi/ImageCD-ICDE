# q5_storage_refresh.py
import argparse, os, time, random
from pathlib import Path
import pandas as pd
from q5_common import ensure_parquet, init_rw_db, load_parquet_as_table, human_bytes

def db_size_bytes(db_path: Path)->int:
    return os.path.getsize(db_path) if db_path.exists() else 0

def simulate_refresh(con, user_ids, concepts_to_update=3)->float:
    t0=time.perf_counter()
    for uid in user_ids:
        con.execute("UPDATE ProficiencyImageView SET proficiency = random() WHERE user_id=? AND concept_id<=?;",
                    [uid, concepts_to_update])
    el=time.perf_counter()-t0
    return len(user_ids)/max(el,1e-9)

def main():
    ap=argparse.ArgumentParser("Q5 Storage & Refresh (small-scale)")
    ap.add_argument("--db-path", type=Path, default=Path("./imag2data_small.duckdb"))
    ap.add_argument("--parquet", type=Path, default=Path("./synthetic_small.parquet"))
    ap.add_argument("--num_users", type=int, default=2000)         # 小规模，避免卡住
    ap.add_argument("--num_concepts", type=int, default=50)
    ap.add_argument("--num_difficulties", type=int, default=25)
    ap.add_argument("--sparsity", type=float, default=1.0)
    ap.add_argument("--refresh-users", type=int, default=500)
    ap.add_argument("--outdir", type=Path, default=Path("./q5_outputs2/storage_refresh"))
    args=ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    ensure_parquet(args.parquet, args.num_users, args.num_concepts, args.num_difficulties, args.sparsity)

    con=init_rw_db(args.db_path)
    load_parquet_as_table(con, args.parquet)

    size=db_size_bytes(args.db_path)
    per_user=size/max(args.num_users,1)

    sample= random.sample(range(1,args.num_users+1), k=min(args.refresh_users, args.num_users))
    thr=simulate_refresh(con, sample, concepts_to_update=3)

    df=pd.DataFrame([{
        "DB Size (bytes)": size,
        "DB Size (human)": human_bytes(size),
        "Per-User (bytes)": per_user,
        "Per-User (KB)": per_user/1024.0,
        "Refresh Throughput (users/s)": thr
    }])
    df.to_csv(args.outdir/"storage_refresh.csv", index=False)
    print(df)

if __name__=="__main__":
    main()
