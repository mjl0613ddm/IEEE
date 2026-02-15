# Plot (EconAgent)

批量生成 Shapley 与风险相关图表。

## 用法

```bash
python scripts/plot/batch_plot.py [--models claude gpt] [--seeds claude_42]
```

## 输出

4 张图保存到 `datas/{model}/{seed}/plot/`：
- 风险曲线与累积 Shapley
- Shapley 绝对值 vs 不稳定性（C_ag）
- 按 Agent 聚合
- 按 Behavior 聚合
