# Plot (SocialLLM)

风险与 Shapley 可视化。

## 脚本

| 脚本 | 用途 |
|------|------|
| `plot_risk.py` | 风险折线图 |
| `plot_risk_and_shapley.py` | 风险曲线 + 累积 Shapley 双子图 |
| `batch_plot_risk.sh` | 批量风险图 |
| `batch_plot_risk_and_shapley.sh` | 批量风险+Shapley 图 |

## 用法

```bash
# 单次
python scripts/plot/plot_risk.py results/gpt-4o-mini/gpt-4o-mini_42
python scripts/plot/plot_risk_and_shapley.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42

# 批量
bash scripts/plot/batch_plot_risk.sh
bash scripts/plot/batch_plot_risk_and_shapley.sh [--models gpt-4o-mini]
```

## 输出

- `results/{model}/{seed}/plot/risk_timeseries.png`
- `results/{model}/{seed}/plot/risk_and_cumulative_shapley.png`
