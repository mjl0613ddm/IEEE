# Risk Features (EconAgent)

计算 5 个风险特征指标：L_tm, G_ag, C_ag, Z_ag, G_be。

## 快速开始

```bash
# 1. 批量计算
python batch_calculate_risk_features.py --max_workers 64

# 2. 汇总
python aggregate_risk_features.py

# 3. 可视化
python visualize_behavior_risk.py
```

## 脚本

| 脚本 | 用途 |
|------|------|
| `calculate_risk_features.py` | 单 seed 计算 |
| `batch_calculate_risk_features.py` | 批量并行计算 |
| `aggregate_risk_features.py` | 汇总结果 |
| `visualize_behavior_risk.py` | G_be 行为风险可视化 |

## 参数

- `--data_dir` / `--datas_root`: 数据根目录
- `--models`: 指定模型（如 `claude gpt`）
- `--max_workers`: 并行数

## 输出

- 结果: `results/risk_features_exp/{model}/`
- 汇总: `all_models_summary.csv`, `{model}_summary.csv`
