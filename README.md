# README

## Project Introduction

This repository consists of three parts:
- Model training and inference: a training and evaluation pipeline composed of `main.py`, `models.py`, and `doa.py`.
- Database benchmark (DB): scripts in the `DB/` directory are used to simulate data storage and indexing operations (DuckDB + Parquet), and evaluate latency/throughput for typical query and refresh scenarios.
- Simulator: DuckDB-based end-to-end simulator that generates synthetic logs, builds RawImageView and ProficiencyImageView with indexes, runs sample queries, and exports cognitive image.

## Directory Structure

- `main.py`: Training entry point, which reads data, builds the generator network, trains, and outputs student proficiency maps; optional reporting to W&B.
- `models.py`: Core module `MultiBlock`, used for multi-scale feature extraction in the generator.
- `doa.py`: Calculates the DOA metric (relative consistency between proficiency ranking and answer correctness), supporting multi-dataset configurations.
- `DB/`: Database benchmark subproject (see details below).
- `Simulator/`: DuckDB-based end-to-end simulator that generates synthetic logs, builds RawImageView and ProficiencyImageView with indexes, runs sample queries, and exports heatmaps.
- `data.7z`, `dataset.7z`, `student_info.7z`: Compressed data packages.

## Simulator: End-to-End DuckDB Pipeline (Quick Start)

To help users quickly understand ImageCD’s full data pipeline and query/indexing workflow without preparing real data, this repository offers a one-click simulator based on DuckDB in the Simulator directory.

It executes the following steps in order (printing stage-wise progress and statistics to the terminal):

- LogView: randomly generates user–item interaction logs (including concept, difficulty, correctness, and timestamp)
- Data2Imag: aggregates into `RawImageView` by (user × concept × difficulty)
- Imag2Diag (simulated): imputes sparse mastery maps, adds small noise, and enforces monotonicity along the difficulty axis to produce a dense mastery image
- Imag2Data: materializes results into the `proficiency_image` table (i.e., ProficiencyImageView) and builds multi-column indexes
- Query demos: point query, concept-wise distribution, difficulty-wise distribution, and saving a small submatrix of mastery values as a heatmap image

Dependencies (CPU-only is fine, decoupled from training):

```bash
pip install -r Simulator/requirements.txt
```

Run the simulator:

```bash
python3 Simulator/sim_pipeline.py
```

Default behavior:
- Generates DuckDB database: simulator.duckdb
- Creates/fills tables: `log_view`, `raw_image_view`, `proficiency_image`
- Builds indexes:
  - `(user_id, concept_id, difficulty_id)`
  - `(user_id, difficulty_id, concept_id)`
- Produces and saves a heatmap image (example: user 0, concepts 0..4, difficulties 0..2): heatmap_u0_c0-4_d0-2.png

Optional parameters (example):

```bash
python3 Simulator/sim_pipeline.py \
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

See README.md for more details. This simulator is suitable for end-to-end pipeline demonstrations and quick validation in papers or system documentation.

## Quick Start (Training Part)

1. Extract the data packages to the current directory:
    - `data.7z`
    - `dataset.7z`  (Fix: original text incorrectly wrote `dataset.z7`)
    - `student_info.7z`

2. Configure data paths:
    - `main.py` uses absolute paths by default, e.g.: `/data/{data_name}`, `/datasets/{data_name}`.
    - You can:
       - Create symbolic links of the extracted directories to the root directory, such as `/data` and `/datasets`; or
       - Directly modify the relevant paths in `main.py` to your local actual paths.

3. Run training:
    ```bash
    nohup python main.py \
       --wandb_info "a2017" --train_val 0.3 --window_size 0 \
       --rates 4 --coder_number 1 --block_number 4 --data_name "a2017" \
       --fk 8 --sk 4 --lr 0.002 --batch_size 32 --epoch 30 --dim 128 \
       --seed 3702 --optim_sche 1 --save_info 1 --loss_w 1.0 --diff_dim 25 \
       > test_a2017.log 2>&1 &
    ```

Note: `environment.txt` in the root directory is a Conda environment definition based on Linux/CUDA. If debugging on the CPU side of macOS only, you can install dependencies by simplifying as needed.


## DB Subproject: Storage & Index Simulation + Benchmarking

In the `DB/` directory, we simulate the storage and indexing of proficiency tables with dimensions "user × concept × difficulty", and evaluate two typical query types and refresh throughput:

- Data model: 3D coordinates `(user_id, concept_id, difficulty_id)` → `proficiency` (floating-point number).
- Storage format: Parquet (Snappy compression), query engine: DuckDB.
- Indexes: After loading Parquet, multi-column indexes are built on the in-memory table `ProficiencyImageView`:
  - `(user_id, concept_id, difficulty_id)`
  - `(user_id, difficulty_id, concept_id)`
- Query scenarios:
  - Point lookup: Retrieve a single value given a triplet (using the full prefix index).
  - Small-range interval aggregation: Fix `user_id`, perform AVG on small windows of `concept_id` and `difficulty_id` using BETWEEN (utilizing index prefixes).
- Refresh scenario: Simulate small-scale UPDATE for some users, and count throughput in users/second.

Required dependencies (decoupled from training, CPU environment only):
```bash
pip install duckdb pyarrow pandas numpy
```

Quick run:

- Batch script:
  ```bash
  bash DB/bash.sh
  ```