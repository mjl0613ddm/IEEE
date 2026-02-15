# Batch Shapley (TwinMarket)

批量计算 Shapley 值归因：对 (user_id, date) 玩家计算风险指标 `risk_indicator_simple` 的 Monte Carlo Shapley 贡献。Baseline 为 hold（被 mask 的玩家不提交订单）。

**纯内存计算，不访问数据库**：直接读取 `trading_records/*.json`，使用 `matching_engine_memory` 做撮合，不依赖 sys_*.db。若存在 `analysis/market_metrics.csv`，会优先读取 `real_metric` 避免重跑全联盟。

## 目录结构要求

- **必需**: `result_dir/trading_records/`，内含 `YYYY-MM-DD.json` 决策文件（不含 `_orders` 后缀的 json）。
- **可选**: `result_dir/simulation_results/` — 若存在，用于推断首日 `last_prices`；否则使用默认价格。
- **可选**: `result_dir/analysis/market_metrics.csv` — 若存在，`batch_calculate_shapley.sh` 可调用 `find_max_risk_date.py` 自动得到 `start_date` / `target_date`。

## 用法

### 单目录

```bash
# 指定结果目录与日期
RESULT_DIR=/path/to/gpt-4o-mini_46 \
START_DATE=2023-06-19 TARGET_DATE=2023-08-11 \
N_SAMPLES=1000 N_THREADS=0 \
bash batch_calculate_shapley.sh
```

若 `result_dir/analysis/market_metrics.csv` 存在，可省略日期，脚本会自动推断：

```bash
RESULT_DIR=/path/to/gpt-4o-mini_46 N_SAMPLES=1000 bash batch_calculate_shapley.sh
```

### 直接调用 Python

```bash
cd /path/to/TwinMarket
python scripts/batch_culculate_shapley/calculate_shapley.py \
  --result_dir /path/to/gpt-4o-mini_46 \
  --start_date 2023-06-19 --target_date 2023-08-11 \
  --n_samples 1000 --seed 42 --n_threads 0
```

## 输出

输出目录默认为 `result_dir/shapley/`，包含：

| 文件 | 说明 |
|------|------|
| `shapley_stats_{start}_{end}.json` | 元信息、baseline_metric、real_metric、n_samples、shapley 统计 |
| `shapley_attribution_timeseries_{start}_{end}.csv` | 每 (user_id, date) 的 shapley_value |
| `shapley_matrix_{start}_{end}.npy` | 矩阵 shape (n_users, n_dates)，供 risk_feature / plot 使用 |
| `shapley_labels_{start}_{end}.npy` | 含 `user_ids`、`dates` 的 dict |
| `baseline_risk_indicators_{start}_{end}.csv` | 按日的 baseline_risk、real_risk |

## 参数

- `--n_samples`: Monte Carlo 排列采样数（默认 1000）
- `--n_threads`: 并行进程数，0 表示使用 CPU 核心数
- `--base_path`: 项目根目录，用于加载 `data/stock_profile.csv` 权重（可选）
