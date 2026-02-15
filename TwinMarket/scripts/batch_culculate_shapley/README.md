# 批量计算Shapley值脚本

## 概述

这个脚本系统用于批量处理TwinMarket实验结果，自动识别每个实验结果文件夹中的最高风险点，并对该风险点之前的所有agents和日期进行Shapley值归因计算。

## 文件说明

- `batch_calculate_shapley.sh`: 主批量处理脚本
- `find_max_risk_date.py`: 辅助Python脚本，用于从market_metrics.csv中查找最高风险点日期

## 使用方法

### 1. 配置参数

编辑 `batch_calculate_shapley.sh` 脚本顶部的配置参数：

```bash
# 模型名称（例如: gpt-4o-mini, deepseek-v3.2）
MODEL_NAME="gpt-4o-mini"

# Shapley计算参数
N_SAMPLES=1000          # 蒙特卡洛采样次数
N_THREADS=0             # 并行线程数（0表示使用CPU核心数）
BASELINE_TYPE="hold"    # Baseline策略类型: hold 或 no_action
SEED=42                 # 随机数种子
```

### 2. 运行脚本

```bash
cd /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/scripts/batch_culculate_shapley
bash batch_calculate_shapley.sh
```

或者直接：

```bash
bash /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/scripts/batch_culculate_shapley/batch_calculate_shapley.sh
```

### 3. 切换模型

只需修改脚本中的 `MODEL_NAME` 变量：

```bash
MODEL_NAME="deepseek-v3.2"  # 从 gpt-4o-mini 改为 deepseek-v3.2
```

## 工作原理

1. **扫描子文件夹**: 脚本会扫描 `results/{MODEL_NAME}/` 目录下的所有子文件夹（如 `gpt-4o-mini_42`, `gpt-4o-mini_43` 等）

2. **识别最高风险点**: 对每个子文件夹，调用 `find_max_risk_date.py` 脚本：
   - 读取 `analysis/market_metrics.csv` 文件
   - 找到 `risk_indicator_simple` 列的最大值对应的日期
   - 返回开始日期（第一行有效日期）和目标日期（最高风险点）

3. **计算Shapley值**: 对每个子文件夹调用 `calculate_shapley_attribution_new.py` 脚本，计算从开始日期到目标日期的Shapley值归因

4. **保存结果**: 计算结果保存在每个子文件夹的 `shapley/` 目录下

## 输出结果

### 每个子文件夹的结果

保存在 `{result_dir}/shapley/` 目录下：
- `shapley_attribution_timeseries_{start_date}_{target_date}.csv`: Shapley值结果CSV
- `baseline_risk_indicators_{start_date}_{target_date}.csv`: Baseline风险指标时间序列
- `shapley_stats_{start_date}_{target_date}.json`: 统计信息JSON
- `shapley_values_{start_date}_{target_date}.npy`: Shapley值npy文件
- `shapley_matrix_{start_date}_{target_date}.npy`: Shapley值矩阵npy文件
- `shapley_labels_{start_date}_{target_date}.npy`: Shapley值标签npy文件

### 脚本执行日志

- 详细日志: `batch_calculate_shapley_{timestamp}.log`
- 汇总报告: `batch_calculate_shapley_summary_{timestamp}.txt`

## 注意事项

1. **依赖要求**: 确保已安装必要的Python包（pandas, numpy等）

2. **磁盘空间**: 批量计算可能需要较长时间和大量磁盘空间，建议：
   - 在后台运行或使用screen/tmux
   - 确保有足够的临时空间

3. **结果覆盖**: 如果某个子文件夹已经存在shapley结果，脚本会覆盖现有结果

4. **错误处理**: 如果某个子文件夹处理失败，脚本会记录错误但继续处理其他文件夹

## 示例

假设有以下文件夹结构：

```
results/gpt-4o-mini/
├── gpt-4o-mini_42/
│   ├── analysis/
│   │   └── market_metrics.csv
│   └── trading_records/
├── gpt-4o-mini_43/
│   ├── analysis/
│   │   └── market_metrics.csv
│   └── trading_records/
└── ...
```

运行脚本后，每个子文件夹下会生成：

```
gpt-4o-mini_42/
├── analysis/
├── trading_records/
└── shapley/
    ├── shapley_attribution_timeseries_2023-06-15_2023-08-02.csv
    ├── shapley_stats_2023-06-15_2023-08-02.json
    └── ...
```
