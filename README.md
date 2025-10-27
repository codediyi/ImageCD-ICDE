# README

## 项目简介

本仓库包含两部分：
- 模型训练与推断：`main.py`、`models.py`、`doa.py` 组成的训练与评估流程。
- 数据库基准（DB）：`DB/` 目录内的脚本用于模拟数据的存储与索引操作（DuckDB + Parquet），并对典型查询与刷新场景进行延迟/吞吐评测。

## 目录结构

- `main.py`：训练入口，读取数据、构建生成器网络、训练并产出学生掌握度图；可选上报到 W&B。
- `models.py`：核心模块 `MultiBlock`，在生成器中用于多尺度特征提取。
- `doa.py`：计算 DOA 指标（掌握度排序与作答正确性的相对一致性），支持多数据集配置。
- `DB/`：数据库基准子项目（详见下文）。
- `data.7z`、`dataset.7z`、`student_info.7z`：数据压缩包。

## 快速开始（训练部分）

1. 解压数据包到当前目录：
    - `data.7z`
    - `dataset.7z`  （修复：原文写成了 `dataset.z7`）
    - `student_info.7z`

2. 配置数据路径：
    - `main.py` 中默认使用绝对路径，例如：`/data/{data_name}`、`/datasets/{data_name}`。
    - 你可以：
       - 将解压后的目录软链到根目录，如 `/data` 和 `/datasets`；或
       - 直接修改 `main.py` 中相关路径为你的本地实际路径。

3. 运行训练：
    ```bash
    nohup python main.py \
       --wandb_info "a2017" --train_val 0.3 --window_size 0 \
       --rates 4 --coder_number 1 --block_number 4 --data_name "a2017" \
       --fk 8 --sk 4 --lr 0.002 --batch_size 32 --epoch 30 --dim 128 \
       --seed 3702 --optim_sche 1 --save_info 1 --loss_w 1.0 --diff_dim 25 \
       > test_a2017.log 2>&1 &
    ```

提示：根目录的 `environment.txt` 是基于 Linux/CUDA 的 Conda 环境定义，若仅在 macOS 上做 CPU 侧调试，可按需自行精简安装依赖。

## DB 子项目：存储与索引模拟 + 基准测试

我们在 `DB/` 目录中模拟了“用户 × 知识点 × 难度”的熟练度表的存储与索引，并评测两类典型查询以及刷新吞吐：

- 数据模型：三维坐标 `(user_id, concept_id, difficulty_id)` → `proficiency`（浮点数）。
- 存储格式：Parquet（Snappy 压缩），查询引擎：DuckDB。
- 索引：载入 Parquet 后在内存表 `ProficiencyImageView` 上建立多列索引：
   - `(user_id, concept_id, difficulty_id)`
   - `(user_id, difficulty_id, concept_id)`
- 查询场景：
   - 点查：给定三元组取单值（走完整前缀索引）。
   - 小范围区间聚合：固定 user_id，对 `concept_id` 与 `difficulty_id` 做 BETWEEN 的小窗口 AVG（利用索引前缀）。
- 刷新场景：模拟对部分用户的小规模 UPDATE，统计用户/秒吞吐。

必要依赖（与训练解耦，仅需 CPU 环境）：
```bash
pip install duckdb pyarrow pandas numpy
```

快速运行：

- 批量脚本：
   ```bash
   bash DB/bash.sh
