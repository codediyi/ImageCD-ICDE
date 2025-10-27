#!/usr/bin/env bash
set -euo pipefail

PY=python

NAMES=("assist2012" "xes3g5m" "assist2017" "algebra2005")
USERS=("27485" "18066" "1708" "574")      
CONCEPTS=("265" "831" "102" "112")        

DIFFS=(10 15 20 25 30)  


for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  U=${USERS[$i]}
  C=${CONCEPTS[$i]}
  for d in "${DIFFS[@]}"; do
    OUTDIR="./q5_outputs2/latency/${name}_D${d}"
    mkdir -p "$OUTDIR"
    ${PY} q5_latency.py \
      --db-path "./db_${name}_D${d}.duckdb" \
      --parquet "./parquet_${name}_D${d}.parquet" \
      --num_users "$U" --num_concepts "$C" --num_difficulties "$d" \
      --sparsity 1.0 --latency-iters 300 \
      --outdir "$OUTDIR" | tee "${OUTDIR}/run.log"
  done
done


for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  U=${USERS[$i]}
  C=${CONCEPTS[$i]}
  for d in "${DIFFS[@]}"; do

    U_SMALL=$(( U < 5000 ? U : 5000 ))
    REFRESH=$(( U_SMALL < 1000 ? U_SMALL : 1000 ))
    OUTDIR="./q5_outputs2/storage_refresh/${name}_D${d}"
    mkdir -p "${OUTDIR}"

    ${PY} q5_storage_refresh.py \
      --db-path "./db_${name}_D${d}_small.duckdb" \
      --parquet "./parquet_${name}_D${d}_small.parquet" \
      --num_users "${U_SMALL}" --num_concepts "${C}" --num_difficulties "${d}" \
      --sparsity 1.0 --refresh-users "${REFRESH}" \
      --outdir "${OUTDIR}" | tee "${OUTDIR}/run.log"
  done
done