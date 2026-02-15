# Batch Shapley (TwinMarket)

批量计算 Shapley 值归因。

## 用法

1. 编辑 `batch_calculate_shapley.sh`：
   ```bash
   MODEL_NAME="gpt-4o-mini"
   N_SAMPLES=1000
   N_THREADS=0  # 0=使用 CPU 核心数
   ```

2. 运行：
   ```bash
   bash batch_calculate_shapley.sh
   ```

## 输出

- `{result_dir}/shapley/shapley_*.npy`, `*.csv`, `*.json`
