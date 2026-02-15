# Risk Features (TwinMarket)

计算 5 个风险特征指标：L_tm, G_ag, C_ag, Z_ag, G_be。

## 快速开始

```bash
# 1. 批量计算
python batch_calculate_risk_features.py --max_workers 64

# 2. 汇总
python aggregate_risk_features.py

# 3. 可视化
python visualize_risk_features.py [--skip_accumulation]
```

## 脚本

| 脚本 | 用途 |
|------|------|
| `calculate_risk_features.py` | 单 seed 计算 |
| `batch_calculate_risk_features.py` | 批量并行计算 |
| `aggregate_risk_features.py` | 汇总 |
| `visualize_risk_features.py` | L_tm 与 G_be 可视化 |

## 输出

- 结果: `results/risk_feature/{model}/`
- 汇总: `all_models_summary.csv`
