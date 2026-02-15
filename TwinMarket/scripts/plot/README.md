# Plot (TwinMarket)

批量生成 Shapley 与风险图表。

## 用法

```bash
python scripts/plot/batch_plot.py [--models gpt-4o-mini] [--seeds gpt-4o-mini_42]
```

## 输出

4 张图保存到 `results/{model}/{seed}/plot/`：
- 风险曲线与累积 Shapley
- Shapley vs 不稳定性
- Agent 聚合
- Behavior 聚合
