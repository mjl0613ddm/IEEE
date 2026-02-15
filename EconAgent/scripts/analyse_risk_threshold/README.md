# Risk Threshold Analysis

分析最高风险值相对第一个有效值的百分比。

## 用法

```bash
python scripts/analyse_risk_threshold/analyse_risk_threshold.py \
    --datas_dir datas --output_dir results/analyse_risk_threshold
```

## 输出

- `risk_threshold_summary.csv`: 按模型汇总
- `risk_threshold_detailed.csv`: 明细
- `risk_threshold_analysis.json`: 完整 JSON
