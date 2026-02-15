# SocialLLM 结果筛选脚本

本目录包含用于筛选可以进行归因的SocialLLM模拟结果的脚本。

## 文件说明

- `filter_by_risk.py`: 主筛选脚本
- `README.md`: 本文档

## 功能概述

根据风险指标筛选可以进行归因的模拟结果，不符合标准的文件夹会被添加 `_rm` 后缀。

### 筛选标准

1. **最高风险时刻 >= 10**：`max_risk_timestep >= 10`
2. **最高风险比初始风险高150%以上**：`max_risk >= initial_risk * 2.5`

不符合任一标准的文件夹会被添加 `_rm` 后缀。

## 使用方法

### 基本用法

```bash
cd SocialLLM  # or path to SocialLLM directory

# 默认模式（实际执行，会重命名文件夹）
python3 scripts/filter/filter_by_risk.py

# 指定结果目录
python3 scripts/filter/filter_by_risk.py --results_dir results/

# 指定输出目录
python3 scripts/filter/filter_by_risk.py --output_dir results/filter_results/
```

### 干运行模式（推荐先使用）

在正式执行前，建议先使用干运行模式检查哪些文件夹会被重命名：

```bash
python3 scripts/filter/filter_by_risk.py --dry_run
```

### 完整参数示例

```bash
python3 scripts/filter/filter_by_risk.py \
    --results_dir results/ \
    --output_dir results/filter_results/ \
    --base_path /path/to/SocialLLM \
    --dry_run
```

## 参数说明

- `--results_dir`: 结果根目录路径（可选，默认：项目根目录下的 `results/`）
- `--output_dir`: 输出目录路径（可选，默认：`results/filter_results/`）
- `--base_path`: 项目根目录路径（可选，默认：自动检测）
- `--dry_run`: 干运行模式（只检查不实际重命名）

## 筛选逻辑

### 标准1：最高风险时刻 >= 10

检查 `results.json` 中的 `max_risk_timestep` 字段：
- 如果 `max_risk_timestep < 10`，则不符合标准

### 标准2：最高风险比初始风险高150%以上

检查 `results.json` 中的 `max_risk` 和 `initial_risk` 字段：
- 计算阈值：`threshold = initial_risk * 2.5`
- 如果 `max_risk < threshold`，则不符合标准

### 示例

**示例1：符合标准**
- `max_risk_timestep = 15`
- `max_risk = 0.607`
- `initial_risk = 0.080`
- 条件1: `15 >= 10` ✓
- 条件2: `0.607 >= 0.080 * 2.5 = 0.200` ✓
- **结果：不需要重命名**

**示例2：不符合标准（时刻太早）**
- `max_risk_timestep = 5`
- `max_risk = 0.607`
- `initial_risk = 0.080`
- 条件1: `5 < 10` ✗
- **结果：需要重命名（添加 `_rm` 后缀）**

**示例3：不符合标准（风险增长不足）**
- `max_risk_timestep = 15`
- `max_risk = 0.150`
- `initial_risk = 0.080`
- 条件1: `15 >= 10` ✓
- 条件2: `0.150 < 0.080 * 2.5 = 0.200` ✗
- **结果：需要重命名（添加 `_rm` 后缀）**

## 输出格式

### 控制台输出

脚本会输出：
- 处理进度（每个模型和结果目录）
- 风险指标信息
- 重命名操作结果
- 汇总统计信息

### 文件输出

在输出目录（默认：`results/filter_results/`）下生成：

1. **每个模型的汇总文件**：
   - `{model}/summary.json`: JSON格式的详细统计信息
   - `{model}/summary.csv`: CSV格式的有效结果列表

2. **总体汇总文件**：
   - `overall_summary.json`: 所有模型的统计信息

### 汇总文件格式

**summary.json** 包含：
- `model`: 模型名称
- `summary`: 统计信息
  - `count`: 有效结果数量
  - `avg_max_risk_timestep`: 平均最大风险步数
  - `median_max_risk_timestep`: 最大风险步数中位数
  - `std_max_risk_timestep`: 最大风险步数标准差
  - `max_risk_value`: 最大风险值
  - `min_risk_value`: 最小风险值
  - `avg_max_risk_value`: 平均最大风险值
  - `median_max_risk_value`: 最大风险值中位数
  - `std_max_risk_value`: 最大风险值标准差
  - `avg_initial_risk`: 平均初始风险值
  - `median_initial_risk`: 初始风险值中位数
- `details`: 每个有效结果的详细信息

**summary.csv** 包含列：
- `result_dir_name`: 结果目录名称
- `max_risk_timestep`: 最高风险时刻
- `max_risk`: 最高风险值
- `initial_risk`: 初始风险值
- `risk_ratio`: 风险比率（max_risk / initial_risk）

## 目录结构

脚本处理的目录结构：
```
results/
├── {model_name}/
│   ├── {model_name}_{seed}/
│   │   └── results.json
│   ├── {model_name}_{seed}_rm/  # 不符合标准的文件夹
│   └── ...
└── filter_results/  # 输出目录
    ├── {model_name}/
    │   ├── summary.json
    │   └── summary.csv
    └── overall_summary.json
```

## 注意事项

1. **备份数据**：在执行前建议备份重要数据，或先使用 `--dry_run` 模式检查

2. **已标记文件夹**：脚本会自动跳过已包含 `_rm` 后缀的文件夹

3. **错误处理**：如果某个结果目录的 `results.json` 文件缺失或格式错误，会记录错误但不会中断处理

4. **重命名操作**：重命名操作是不可逆的，请谨慎使用

5. **并发安全**：如果多个进程同时运行脚本，可能会产生冲突，建议串行执行

## 依赖要求

- Python 3.7+
- 依赖包：
  - `pandas`
  - `numpy`

## 示例输出

```
================================================================================
SocialLLM 结果筛选脚本
================================================================================
结果目录: /path/to/results
输出目录: /path/to/results/filter_results
模式: 实际执行（会重命名文件夹）
================================================================================

处理模型: gpt-4o-mini
--------------------------------------------------------------------------------
  找到 15 个结果目录

处理: gpt-4o-mini_42
    最高风险时刻: 15
    最高风险值: 0.607805
    初始风险值: 0.080390
    风险阈值: 0.200975
  - 无需重命名: gpt-4o-mini_42
    原因: 符合筛选标准

处理: gpt-4o-mini_43
    最高风险时刻: 5
    最高风险值: 0.500000
    初始风险值: 0.080000
    风险阈值: 0.200000
  ✓ 已重命名: gpt-4o-mini_43 -> gpt-4o-mini_43_rm
    原因: 不符合筛选标准

...

模型 gpt-4o-mini:
  未被标记的文件夹数量: 12
  平均最大风险步数: 14.25
  最大风险步数中位数: 15.00
  ...
```
