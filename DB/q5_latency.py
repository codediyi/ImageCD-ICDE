# q5_latency.py
import argparse, time, random
from pathlib import Path
import pandas as pd
import sys
from q5_common import ensure_parquet, init_rw_db, load_parquet_as_table, percentile

def bench_point(con, iters, U, C, D, seed=7):
    rng = random.Random(seed); lat=[]
    for _ in range(iters):
        uid = rng.randint(1,U); cid=rng.randint(1,C); did=rng.randint(1,D)
        t0=time.perf_counter()
        con.execute("SELECT proficiency FROM ProficiencyImageView WHERE user_id=? AND concept_id=? AND difficulty_id=?",
                    [uid,cid,did]).fetchone()
        lat.append((time.perf_counter()-t0)*1000)
    return percentile(lat)

def bench_range(con, iters, U, C, D, seed=8):
    rng = random.Random(seed); lat=[]
    for _ in range(iters):
        uid=rng.randint(1,U)
        i1=rng.randint(1,max(1,C-6)); i2=i1+rng.randint(0,6)
        j1=rng.randint(1,max(1,D-6)); j2=j1+rng.randint(0,6)
        t0=time.perf_counter()
        con.execute("""SELECT AVG(proficiency) FROM ProficiencyImageView
                       WHERE user_id=? AND concept_id BETWEEN ? AND ? AND difficulty_id BETWEEN ? AND ?""",
                    [uid,i1,i2,j1,j2]).fetchone()
        lat.append((time.perf_counter()-t0)*1000)
    return percentile(lat)

def main():
    ap=argparse.ArgumentParser("Q5 Latency")
    ap.add_argument("--db-path", type=Path, default=Path("./imag2data.duckdb"))
    ap.add_argument("--parquet", type=Path, default=Path("./synthetic.parquet"))
    ap.add_argument("--num_users", type=int, default=10000)
    ap.add_argument("--num_concepts", type=int, default=50)
    ap.add_argument("--num_difficulties", type=int, default=25)
    ap.add_argument("--sparsity", type=float, default=1.0)
    ap.add_argument("--latency-iters", type=int, default=300)
    ap.add_argument("--outdir", type=Path, default=Path("./q5_outputs2/latency"))
    args=ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    ensure_parquet(args.parquet, args.num_users, args.num_concepts, args.num_difficulties, args.sparsity)
    con=init_rw_db(args.db_path); load_parquet_as_table(con, args.parquet)

    p50,p95,p99=bench_point(con,args.latency_iters,args.num_users,args.num_concepts,args.num_difficulties)
    r50,r95,r99=bench_range(con,args.latency_iters,args.num_users,args.num_concepts,args.num_difficulties)

    df=pd.DataFrame([
        ("Point Lookup",p50,p95,p99),
        ("Range (no prefix)",r50,r95,r99),
    ], columns=["Query Type","P50 (ms)","P95 (ms)","P99 (ms)"])
    df.to_csv(args.outdir/"latency.csv", index=False)
    print(df)

if __name__=="__main__":
    main()
