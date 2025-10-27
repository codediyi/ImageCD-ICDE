# ImageCD-ICDE Simulator (DuckDB)

This simulator implements the full pipeline with synthetic data in DuckDB:

1. LogView (raw interaction logs)
2. Data2Imag → RawImageView (per-user concept×difficulty aggregation)
3. Imag2Diag (simulated with noise + monotonic constraint)
4. Imag2Data → materialized table `proficiency_image`
5. Query demos (point, concept-level, difficulty-level, range aggregation)

The goal is to mimic the end-to-end data processing and indexing flow described in the paper using a lightweight, reproducible setup.

## Quick start

### 1) Create a virtual environment (optional but recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the simulator

```bash
python sim_pipeline.py
```

This will:
- Generate synthetic logs for 50 users, 12 concepts, 8 difficulty bins
- Create a DuckDB database at `Simulator/simulator.duckdb`
- Build `log_view`, `raw_image_view`, and `proficiency_image` (materialized)
- Create multi-column indexes for efficient queries
- Execute and print sample queries

### 4) Parameters

```bash
python sim_pipeline.py \
  --users 100 \
  --concepts 16 \
  --difficulties 10 \
  --items-per-cd 30 \
  --sparsity 0.65 \
  --attempts-low 1 \
  --attempts-high 6 \
  --noise-std 0.08 \
  --seed 42
```

- `--db` can be used to specify a custom DuckDB database path.

### 5) Outputs

- DuckDB file: `simulator.duckdb`
- Tables:
  - `log_view(user_id, item_id, concept_id, difficulty_id, correctness, timestamp)`
  - `raw_image_view(user_id, concept_id, difficulty_id, attempts, corrects, accuracy)`
  - `proficiency_image(user_id, concept_id, difficulty_id, mastery_value)`

### 6) Notes on Imag2Diag approximation

- The CNN encoder–decoder is approximated here by imputing missing cells, adding small Gaussian noise, and enforcing monotonic non-increasing mastery along the difficulty axis.
- This is sufficient for demonstrating the end-to-end storage and query pipeline in DuckDB.

### 7) Example queries (in SQL)

- Point query:

```sql
SELECT * FROM proficiency_image
WHERE user_id = 0 AND concept_id = 0 AND difficulty_id = 0;
```

- Concept-level distribution (fixed concept across difficulties):

```sql
SELECT difficulty_id, mastery_value
FROM proficiency_image
WHERE user_id = 0 AND concept_id = 0
ORDER BY difficulty_id;
```

- Difficulty-level distribution (fixed difficulty across concepts):

```sql
SELECT concept_id, mastery_value
FROM proficiency_image
WHERE user_id = 0 AND difficulty_id = 0
ORDER BY concept_id;
```

- Range aggregation:

```sql
SELECT AVG(mastery_value) AS avg_mastery
FROM proficiency_image
WHERE user_id = 0 AND concept_id BETWEEN 0 AND 4 AND difficulty_id BETWEEN 0 AND 2;
```
