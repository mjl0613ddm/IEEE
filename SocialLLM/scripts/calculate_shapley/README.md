# Shapley (SocialLLM)

蒙特卡洛计算 Shapley 值归因。

## 用法

```bash
# 单次
python scripts/calculate_shapley/calculate_shapley.py \
    --result_dir results/gpt-4o-mini/gpt-4o-mini_42 \
    --n_samples 1000 --n_threads 64

# 批量（编辑 batch_calculate_shapley.sh 中 MODEL_NAMES 指定模型）
bash scripts/calculate_shapley/batch_calculate_shapley.sh
```

## 参数

- `--n_samples`: 采样次数（默认 1000）
- `--n_threads`: 并行数（0=CPU 核心数）

## 输出

- `{result_dir}/shapley/shapley_attribution_timeseries.csv`
- `{result_dir}/shapley/shapley_stats.json`
