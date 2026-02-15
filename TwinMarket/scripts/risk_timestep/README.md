# 计算TwinMarket风险最高点平均时间步

该脚本用于计算TwinMarket中各个模型的风险最高点平均时间步。

## 功能说明

1. **计算风险指标**：从每个结果文件夹的 `analysis/market_metrics.csv` 中读取 `risk_indicator_simple` 数据
2. **找到风险最高点**：确定每个结果中风险指标的最大值对应的日期
3. **转换为时间步**：将日期按顺序转换为时间步（从0开始）
4. **汇总统计**：计算所有不带 `_rm` 后缀的结果的平均风险最高点时间步

## 使用方法

### 基本用法

```bash
python calculate_risk_timestep.py
```

### 指定参数

```bash
python calculate_risk_timestep.py \
    --results_dir /path/to/results \
    --output_dir /path/to/output \
    --models claude-3-haiku-20240307 deepseek-v3.2 gpt-4o-mini
```

### 参数说明

- `--results_dir`: 结果根目录路径（默认：`TwinMarket/results`）
- `--output_dir`: 输出目录路径（默认：`TwinMarket/results/risk_timestep_summary`）
- `--models`: 要处理的模型列表（默认：所有5个模型）
  - `claude-3-haiku-20240307`
  - `deepseek-v3.2`
  - `gpt-4o-mini`
  - `llama-3.1-70b-instruct`
  - `qwen-plus`

## 输出文件

脚本会在输出目录下生成以下文件：

### 总体汇总
- `overall_summary.json`: 所有模型的汇总统计信息

### 每个模型的详细汇总
- `{model}/summary.json`: 包含统计信息和详细结果
- `{model}/summary.csv`: CSV格式的详细结果

### 汇总统计信息包括

- `count`: 有效的未被标记的文件夹数量
- `avg_max_risk_timestep`: 平均最大风险时间步
- `median_max_risk_timestep`: 最大风险时间步中位数
- `std_max_risk_timestep`: 最大风险时间步标准差
- `max_risk_value`: 最大风险值
- `min_risk_value`: 最小风险值
- `avg_max_risk_value`: 平均最大风险值
- `median_max_risk_value`: 最大风险值中位数
- `std_max_risk_value`: 最大风险值标准差

## 注意事项

1. 脚本会自动跳过带有 `_rm` 后缀的文件夹
2. 如果某个结果文件夹缺少 `analysis/market_metrics.csv` 文件或缺少必要的列，会在错误信息中记录
3. 时间步从0开始计数，对应第一个有效日期

## 示例输出

```
处理模型: claude-3-haiku-20240307
--------------------------------------------------------------------------------
找到 5 个子文件夹（已跳过标记为_rm的文件夹）

处理: claude-3-haiku-20240307_42
  ✓ 最大风险时间步: 15, 最大风险值: 0.032067, 日期: 2023-07-14

模型 claude-3-haiku-20240307:
  未被标记的文件夹数量: 5
  平均最大风险时间步: 18.40
  最大风险时间步中位数: 17.0
  ...
```
