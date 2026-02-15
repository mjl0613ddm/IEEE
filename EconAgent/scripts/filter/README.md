# Filter (EconAgent)

根据风险指标筛选可归因的轨迹，不符合的文件夹添加 `_rm` 后缀。

## 标准

- 最高风险时刻 > 14
- 基于 RiskMetrics：`h_t = λ·h_{t-1} + (1-λ)·e²_{t-1}`，λ=0.94

## 用法

```bash
python scripts/filter/filter_by_risk.py [--data_dir datas] [--output_dir results/filter_results]
```

## 输出

- 标记: 不符合的文件夹重命名为 `*_rm`
- 报告: `results/filter_results/{model}/summary.json`
