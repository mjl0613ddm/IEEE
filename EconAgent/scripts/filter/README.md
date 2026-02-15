# 数据过滤脚本：根据风险指标标记文件夹

## 功能概述

该脚本用于处理 `ACL24-EconAgent/datas` 目录下的数据，根据 `risk_indicator_naive` 的风险最高点位置自动标记文件夹。

### 主要功能

1. **计算风险指标**：对每个子文件夹计算 `risk_indicator_naive`（基于Engle (1982)和Bollerslev (1986)的风险指标）
2. **自动标记**：如果最高风险点在步骤14及之前（timestep <= 14），自动给文件夹添加 `_rm` 后缀
3. **汇总统计**：按模型汇总未被标记文件夹的统计信息，生成JSON和CSV报告

## 风险指标计算方法

脚本使用 **naive forecast** 方法计算风险指标：

- **公式**：`h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2`
- **参数**：λ = 0.94（RiskMetrics标准）
- **预期规则**：`E_{t-1}[π_t] = π_{t-1}`（naive forecast）
- **预测误差**：`e_t = π_t - E_{t-1}[π_t]`
- **通胀率**：`π_t = log P_t - log P_{t-1}`

其中，`P_t` 是从 `world_metrics.csv` 中的 `price` 列获取的价格数据。

## 使用方法

### 基本用法

```bash
python3 scripts/filter/filter_by_risk.py
```

### 自定义参数

```bash
python3 scripts/filter/filter_by_risk.py \
    --data_dir /path/to/datas \
    --output_dir /path/to/ACL24-EconAgent/results/filter_results
```

### 参数说明

- `--data_dir`: 数据根目录路径（默认：`ACL24-EconAgent/datas`）
- `--output_dir`: 输出目录路径（默认：`ACL24-EconAgent/results/filter_results`）

## 输入数据结构

脚本期望的数据目录结构：

```
datas/
├── claude/
│   ├── claude_42/
│   │   └── metrics_csv/
│   │       └── world_metrics.csv
│   ├── claude_43/
│   └── ...
├── ds/
├── gpt/
├── llama/
└── qwen/
```

每个子文件夹必须包含：
- `metrics_csv/world_metrics.csv`：包含 `timestep` 和 `price` 列的CSV文件

## 输出结果

### 1. 文件夹标记

如果某个子文件夹的最大风险点在步骤14及之前，脚本会自动将其重命名：

- 原始名称：`claude_42`
- 标记后：`claude_42_rm`

**注意**：
- 已包含 `_rm` 后缀的文件夹会被自动跳过，不会重复处理
- 重命名操作会实际修改文件夹名称，请确保有写权限

### 2. 汇总统计报告

脚本会在 `output_dir` 下按模型创建汇总报告：

```
ACL24-EconAgent/results/filter_results/
├── claude/
│   ├── summary.json    # JSON格式的详细统计信息
│   └── summary.csv     # CSV格式的简要统计信息
├── ds/
├── gpt/
├── llama/
├── qwen/
└── overall_summary.json  # 所有模型的总体汇总
```

### JSON报告内容

每个模型的 `summary.json` 包含：

```json
{
  "model": "claude",
  "summary": {
    "count": 10,
    "avg_max_risk_step": 25.5,
    "median_max_risk_step": 24.0,
    "std_max_risk_step": 5.2,
    "max_risk_value": 0.001234,
    "min_risk_value": 0.000456,
    "avg_max_risk_value": 0.000890,
    "median_max_risk_value": 0.000875,
    "std_max_risk_value": 0.000123
  },
  "details": [
    {
      "should_rename": false,
      "max_risk_step": 25,
      "max_risk_value": 0.001234,
      "model": "claude",
      "subfolder_name": "claude_42",
      "original_path": "/path/to/claude/claude_42",
      "error": null
    },
    ...
  ]
}
```

### CSV报告内容

每个模型的 `summary.csv` 包含：

| subfolder_name | max_risk_step | max_risk_value |
|----------------|---------------|----------------|
| claude_42      | 25            | 0.001234       |
| claude_43      | 30            | 0.000987       |
| ...            | ...           | ...            |

**注意**：CSV报告只包含未被标记（`should_rename=False`）的文件夹。

### 统计指标说明

- `count`: 未被标记的文件夹数量
- `avg_max_risk_step`: 平均最大风险步数
- `median_max_risk_step`: 最大风险步数的中位数
- `std_max_risk_step`: 最大风险步数的标准差
- `max_risk_value`: 所有文件夹中的最大风险值
- `min_risk_value`: 所有文件夹中的最小风险值
- `avg_max_risk_value`: 平均最大风险值
- `median_max_risk_value`: 最大风险值的中位数
- `std_max_risk_value`: 最大风险值的标准差

## 处理逻辑

1. **扫描模型目录**：遍历 `claude`, `ds`, `gpt`, `llama`, `qwen` 五个模型目录
2. **跳过已标记文件夹**：自动跳过已包含 `_rm` 后缀的文件夹
3. **读取数据**：从每个子文件夹的 `metrics_csv/world_metrics.csv` 读取价格数据
4. **计算风险指标**：使用 naive forecast 方法计算 `risk_indicator_naive`
5. **找到最大风险点**：过滤掉 NaN 值，找到有效风险值中的最大值及其对应的 timestep
6. **判断是否标记**：如果 `max_risk_step <= 14`，标记文件夹（添加 `_rm` 后缀）
7. **生成统计报告**：按模型汇总未被标记文件夹的统计信息

## 错误处理

脚本具有完善的错误处理机制：

- **CSV文件缺失**：跳过该文件夹，记录错误信息
- **必需的列缺失**：跳过该文件夹，记录错误信息
- **所有风险值为NaN**：跳过该文件夹，记录错误信息
- **计算错误**：跳过该文件夹，记录错误信息，继续处理其他文件夹
- **目标路径已存在**：跳过重命名，记录警告信息

所有错误信息都会记录在JSON报告的 `details` 字段中。

## 注意事项

1. **数据备份**：脚本会实际修改文件夹名称，建议在运行前备份数据
2. **权限要求**：确保对数据目录和输出目录有读写权限
3. **依赖库**：需要安装 `pandas` 和 `numpy`
4. **重复运行**：可以安全地重复运行脚本，已标记的文件夹会被自动跳过

## 示例输出

```
================================================================================
数据过滤脚本：根据risk_indicator_naive标记文件夹
================================================================================
数据目录: /path/to/ACL24-EconAgent/datas
输出目录: /path/to/ACL24-EconAgent/results/filter_results
================================================================================

================================================================================
开始处理所有模型数据
================================================================================

处理模型: claude
--------------------------------------------------------------------------------
找到 5 个子文件夹（已跳过标记为_rm的文件夹）

处理: claude_42
  - 无需重命名: claude_42
    最大风险步数: 25, 最大风险值: 0.001234

处理: claude_43
  ✓ 已重命名: claude_43 -> claude_43_rm
    最大风险步数: 12, 最大风险值: 0.000987

...

================================================================================
生成汇总统计信息
================================================================================

模型 claude:
  未被标记的文件夹数量: 4
  平均最大风险步数: 26.50
  最大风险步数中位数: 25.00
  ...
  ✓ JSON汇总已保存: /path/to/ACL24-EconAgent/results/filter_results/claude/summary.json
  ✓ CSV汇总已保存: /path/to/ACL24-EconAgent/results/filter_results/claude/summary.csv

...

================================================================================
处理完成！
================================================================================
```

## 相关文件

- **风险计算参考**：`scripts/plot/plot_world_metrics.py`
- **数据目录**：`datas/`
- **输出目录**：`results/filter_results/`
