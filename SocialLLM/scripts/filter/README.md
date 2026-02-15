# Filter (SocialLLM)

筛选可归因轨迹，不符合的文件夹添加 `_rm` 后缀。

## 标准

- `max_risk_timestep >= 10`
- `max_risk >= initial_risk * 2.5`

## 用法

```bash
# 推荐先干跑
python scripts/filter/filter_by_risk.py --dry_run

# 正式执行
python scripts/filter/filter_by_risk.py [--results_dir results/] [--output_dir results/filter_results/]
```

## 输出

- 标记: 不符合的文件夹重命名为 `*_rm`
- 报告: `results/filter_results/{model}/summary.json`
