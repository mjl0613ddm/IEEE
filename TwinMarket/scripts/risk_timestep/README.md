# Risk Timestep (TwinMarket)

计算各模型风险最高点的平均时间步。

## 用法

```bash
python calculate_risk_timestep.py [--results_dir results] [--models gpt-4o-mini deepseek-v3.2]
```

## 输出

- `overall_summary.json`: 汇总
- `{model}/summary.json`: 单模型统计
