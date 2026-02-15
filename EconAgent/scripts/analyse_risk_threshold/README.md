# Risk Threshold Analysis Script

## 功能说明

这个脚本用于分析每个模型下不带`rm`后缀的risk indicator naive结果，计算最高值相对于第一个有效值的百分比。

## 使用方法

```bash
python3 scripts/analyse_risk_threshold/analyse_risk_threshold.py \
    --datas_dir /path/to/datas \
    --output_dir /path/to/output
```

### 参数说明

- `--datas_dir`: datas目录路径（默认：`/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/datas`）
- `--output_dir`: 结果输出目录路径（默认：`/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/results/analyse_risk_threshold`）

## 输出结果

脚本会在输出目录生成以下文件：

1. **risk_threshold_summary.csv**: 按模型分类的汇总统计（均值、最大值、最小值、标准差）
2. **risk_threshold_detailed.csv**: 每个种子的详细数据
3. **risk_threshold_analysis.json**: 完整的JSON格式结果（包含所有详细信息）

## 分析逻辑

1. 遍历`datas`目录下的所有模型（如claude, ds, gpt, llama, qwen）
2. 对于每个模型，找到所有不带`_rm`后缀的种子目录
3. 对于每个种子：
   - 从`metrics_csv/world_metrics.csv`读取数据
   - 计算或读取`risk_indicator_naive`值
   - 找到第一个有效的（非NaN且>0）risk indicator naive值
   - 找到从第一个有效值开始的最高值
   - 计算比率：`(最高值 / 第一个值) * 100`
   - **过滤掉比率超过10000%的异常数据**
4. 按模型汇总统计（均值、最大值、最小值、标准差）
5. 最后汇总所有模型的结果

## 数据过滤

脚本会自动过滤掉以下数据：
- 比率超过10000%的异常数据（会在控制台输出警告信息）
- 无法计算比率的数据（如第一个值为0或没有有效值的数据）

## 依赖

- Python 3
- pandas
- numpy

## 示例输出

```
【By Model Summary】
Model           Count    Mean (%)     Max (%)      Min (%)      Std          
claude          10       245.32       320.15       180.45       45.23
gpt             8        220.18       285.67       165.32       38.91
...

【Overall Summary (All Models)】
Total Count: 50
Mean: 235.67%
Max: 350.23%
Min: 150.12%
Std: 42.18
```
