# 风险可视化脚本

## 简介

本目录包含四个脚本：

1. **plot_risk.py**：绘制单个结果目录的风险折线图
2. **batch_plot_risk.sh**：批量绘制所有模型结果的风险折线图
3. **plot_risk_and_shapley.py**：绘制单个结果目录的风险曲线和累积Shapley值图（新增）
4. **batch_plot_risk_and_shapley.sh**：批量绘制所有模型结果的风险曲线和累积Shapley值图（新增）

`plot_risk.py` 用于绘制SocialLLM模拟结果的风险折线图。它会读取每个结果目录中的 `results.json` 文件，绘制风险值随时间步变化的折线图。

`plot_risk_and_shapley.py` 用于绘制风险曲线和累积Shapley值的双子图，参考TwinMarket的`batch_plot.py`实现。

## 功能特性

- 绘制风险时间序列折线图
- 标记初始风险值
- 标记最高风险点和对应的时间步
- 保存为高质量PNG图片（300 DPI）
- 自动创建plot目录

## 安装依赖

首先需要安装matplotlib：

```bash
pip install matplotlib numpy
```

## 使用方法

### 单个结果目录

```bash
cd SocialLLM  # or path to SocialLLM directory
python3 scripts/plot/plot_risk.py results/claude-3-haiku-20240307/claude-3-haiku-20240307_42
```

### 批量绘制（推荐）

使用批量绘制脚本自动识别并绘制所有结果：

```bash
cd SocialLLM  # or path to SocialLLM directory
bash scripts/plot/batch_plot_risk.sh
```

该脚本会：
- 自动识别 `results/` 目录下的所有模型目录
- 自动识别每个模型目录下的所有结果目录（包含 `results.json` 的目录）
- 为每个结果目录绘制风险折线图
- 记录详细的日志信息

### 手动批量绘制

也可以使用bash循环手动批量绘制：

```bash
cd SocialLLM  # or path to SocialLLM directory

# 绘制所有结果
for model_dir in results/*/; do
    for result_dir in "$model_dir"*/; do
        if [ -f "$result_dir/results.json" ]; then
            echo "绘制: $result_dir"
            python3 scripts/plot/plot_risk.py "$result_dir"
        fi
    done
done
```

## 输出

脚本会在结果目录下创建 `plot/` 文件夹，并在其中保存 `risk_timeseries.png` 文件。

例如，对于结果目录：
```
results/claude-3-haiku-20240307/claude-3-haiku-20240307_42/
```

输出文件为：
```
results/claude-3-haiku-20240307/claude-3-haiku-20240307_42/plot/risk_timeseries.png
```

## 图表内容

生成的图表包含：

1. **风险折线图**：显示每个时间步的风险值变化
2. **初始风险线**：绿色虚线标记初始风险值（timestep 0）
3. **最高风险点**：红色星形标记最高风险值和对应的时间步
4. **标题和标签**：包含模型名称、seed、agents数量、steps数量等信息
5. **图例**：显示各个元素的含义

## 参数说明

脚本接受一个参数：

- `results_dir`: 结果目录路径（包含results.json的目录）
  - 可以是绝对路径或相对路径
  - 如果是相对路径，相对于项目根目录

## 示例输出

```
风险折线图已保存到: results/claude-3-haiku-20240307/claude-3-haiku-20240307_42/plot/risk_timeseries.png
绘图完成!
```

## 错误处理

- 如果结果目录不存在，脚本会报错并退出
- 如果 `results.json` 文件不存在，脚本会报错并退出
- 如果 `timestep_results` 为空，脚本会报错并退出
- 所有错误信息会输出到stderr

## 注意事项

1. **依赖**：需要安装matplotlib和numpy
2. **路径**：可以使用绝对路径或相对路径（相对于项目根目录）
3. **权限**：确保有写入权限创建plot目录
4. **文件格式**：输出为PNG格式，分辨率300 DPI

## 批量绘制脚本

### batch_plot_risk.sh

批量绘制脚本 `batch_plot_risk.sh` 用于自动识别并绘制所有模型结果的风险折线图。

#### 使用方法

```bash
cd SocialLLM  # or path to SocialLLM directory
bash scripts/plot/batch_plot_risk.sh
```

#### 功能特性

- 自动识别 `results/` 目录下的所有模型目录
- 自动识别每个模型目录下的所有结果目录
- 跳过不包含 `results.json` 的目录
- 记录详细的日志信息（保存到 `scripts/plot/logs/`）
- 统计成功和失败的数量

#### 日志文件

日志文件保存在 `scripts/plot/logs/` 目录下：

```
logs/
└── batch_plot_risk_20260112_112630.log
```

日志包含：
- 每个任务的启动和完成时间
- 成功/失败状态
- 错误信息（如果有）
- 总体统计信息

#### 输出示例

```
[2026-01-12 11:26:30] ==========================================
[2026-01-12 11:26:30] 批量绘制风险折线图
[2026-01-12 11:26:30] ==========================================
[2026-01-12 11:26:30] 开始时间: 2026-01-12 11:26:30
[2026-01-12 11:26:30] 配置:
[2026-01-12 11:26:30]   结果目录: /path/to/results
[2026-01-12 11:26:30]   绘图脚本: /path/to/scripts/plot/plot_risk.py
[2026-01-12 11:26:30]   日志文件: /path/to/logs/batch_plot_risk_20260112_112630.log
[2026-01-12 11:26:30] 
[2026-01-12 11:26:30] 找到 5 个模型目录
[2026-01-12 11:26:30] 
[2026-01-12 11:26:30] ==========================================
[2026-01-12 11:26:30] 处理模型: claude-3-haiku-20240307
[2026-01-12 11:26:30] ==========================================
[2026-01-12 11:26:30]   找到 15 个结果目录
[2026-01-12 11:26:30]   绘制: claude-3-haiku-20240307_42
[2026-01-12 11:26:30]     ✓ claude-3-haiku-20240307_42 完成 (耗时: 2秒)
...
```

## 3. plot_risk_and_shapley.py - 绘制风险曲线和累积Shapley值图

### 功能说明

`plot_risk_and_shapley.py` 用于绘制风险曲线和累积Shapley值的双子图，参考TwinMarket的`batch_plot.py`实现。

**上子图**：
- 风险演化曲线（红色）
- 累积Shapley值曲线（蓝色）
- 90%最高风险线（橙色虚线）
- 最高风险线（橙色实线）

**下子图**：
- 按时间聚合的Shapley值柱状图
- 正值用红色，负值用蓝色

### 使用方法

```bash
# 单个结果目录
python3 scripts/plot/plot_risk_and_shapley.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42
```

### 参数说明

- `--result_dir`: 结果目录路径（必需）
  - 示例: `results/gpt-4o-mini/gpt-4o-mini_42`
  - 相对路径相对于项目根目录

### 输入文件要求

结果目录必须包含以下文件：

1. **results.json** - 模拟结果文件
   - 包含 `timestep_results`（风险时间序列）
   - 包含 `initial_risk`、`max_risk`、`max_risk_timestep`

2. **shapley/shapley_attribution_timeseries.csv** - Shapley值数据
   - 列: `agent_id`, `timestep`, `shapley_value`

### 输出文件

- `{result_dir}/plot/risk_and_cumulative_shapley.png` - 生成的图表文件

### 注意事项

- 如果输出目录 `plot/` 不存在，脚本会自动创建
- 图表使用300 DPI保存，适合高质量输出
- 累积Shapley值如果超过90%最高风险，会在该点截断

---

## 4. batch_plot_risk_and_shapley.sh - 批量绘制风险曲线和累积Shapley值图

### 功能说明

`batch_plot_risk_and_shapley.sh` 用于批量处理多个模型和种子的风险曲线和累积Shapley值图。

### 使用方法

```bash
# 处理所有模型和种子
bash scripts/plot/batch_plot_risk_and_shapley.sh

# 处理指定的模型
bash scripts/plot/batch_plot_risk_and_shapley.sh --models gpt-4o-mini llama-3.1-8b-instruct
```

### 参数说明

- `--models MODEL1 MODEL2 ...`: 指定要处理的模型列表（可选）
  - 如果不指定，会自动发现`results/`目录下的所有模型
  - 自动排除 `faithfulness_exp` 和 `risk_feature` 目录

### 处理逻辑

1. **自动发现模型**：如果不指定`--models`，脚本会自动扫描`results/`目录
2. **排除规则**：
   - 跳过以`_rm`结尾的种子目录
   - 跳过缺少必要文件的目录（`results.json`或`shapley/shapley_attribution_timeseries.csv`）
   - 跳过已存在输出文件的目录（避免重复处理）
3. **日志记录**：所有输出记录到`scripts/plot/logs/batch_plot_risk_and_shapley_YYYYMMDD_HHMMSS.log`

### 输出信息

脚本会输出以下统计信息：

- 处理的模型数量
- 每个模型处理的种子数量
- 成功/失败的数量
- 详细日志文件路径

### 示例输出

```
[2026-01-14 12:00:00] ==========================================
[2026-01-14 12:00:00] 批量绘制风险曲线和累积Shapley值图
[2026-01-14 12:00:00] ==========================================
[2026-01-14 12:00:00] 处理模型: gpt-4o-mini
[2026-01-14 12:00:00]   处理: gpt-4o-mini_42
[2026-01-14 12:00:00]     ✓ 成功
[2026-01-14 12:00:00] 总计: 成功 15 个, 失败 0 个
```

---

## 相关文件

- `plot_risk.py`: 单个结果目录的风险折线图绘图脚本
- `batch_plot_risk.sh`: 批量绘制风险折线图脚本
- `plot_risk_and_shapley.py`: 单个结果目录的风险曲线和累积Shapley值图绘图脚本（新增）
- `batch_plot_risk_and_shapley.sh`: 批量绘制风险曲线和累积Shapley值图脚本（新增）
- `results.json`: 模拟结果文件（包含timestep_results数组）
- `shapley/shapley_attribution_timeseries.csv`: Shapley值数据文件
- `plot/risk_timeseries.png`: 风险折线图文件
- `plot/risk_and_cumulative_shapley.png`: 风险曲线和累积Shapley值图文件（新增）
- `logs/batch_plot_risk_*.log`: 批量绘制风险折线图的日志文件
- `logs/batch_plot_risk_and_shapley_*.log`: 批量绘制风险曲线和累积Shapley值图的日志文件（新增）

## 更新日志

- 2026-01-12: 初始版本
  - 支持绘制风险时间序列折线图
  - 标记初始风险和最高风险点
  - 自动创建plot目录
  - 添加批量绘制脚本
- 2026-01-14: 新增风险曲线和累积Shapley值图功能
  - 添加 `plot_risk_and_shapley.py` 脚本
  - 添加 `batch_plot_risk_and_shapley.sh` 批量脚本
  - 支持绘制风险曲线和累积Shapley值的双子图