# SocialLLM Shapley Value 计算脚本

本目录包含用于计算 SocialLLM 模拟结果中每个 agent 在每个时间步对最高风险点的 Shapley Value 贡献度的脚本。

## 文件说明

- `calculate_shapley.py`: Shapley Value 计算主脚本
- `batch_calculate_shapley.sh`: 批量计算脚本
- `README.md`: 本文档

## 功能概述

### Shapley Value 计算

使用蒙特卡洛方法计算每个 agent 在每个时间步对最高风险点的贡献度。

**关键特性**：
- **目标指标**：风险最高点的时刻（从 `results.json` 中的 `max_risk_timestep` 自动识别）
- **归因对象**：从 timestep 0 到 `max_risk_timestep-1`，每个 agent 在每个时间步的 Shapley value
- **Baseline 策略**：不行动（masked agent 不发帖、不互动，且该 agent 发的帖子从其他 agent 的视野中移除）
- **计算方法**：蒙特卡洛方法，默认 1000 次采样
- **结果保存**：保存在 `{result_dir}/shapley/` 目录下

## 使用方法

### 单个结果目录计算

```bash
cd SocialLLM  # or path to SocialLLM directory

python3 scripts/calculate_shapley/calculate_shapley.py \
    --result_dir results/gpt-4o-mini/gpt-4o-mini_42 \
    --n_samples 1000 \
    --n_threads 0 \
    --seed 42 \
    --verbose
```

**参数说明**：
- `--result_dir`: 结果目录路径（必须包含 `results.json`, `actions.json`, `random_states.json`）
- `--n_samples`: 蒙特卡洛采样次数（默认：1000）
- `--n_threads`: 并行线程数（默认：1，设为 0 则使用 CPU 核心数，建议设置为实际 CPU 核心数，如 64）
- `--seed`: 随机数种子（默认：42）
- `--verbose`: 输出详细信息（可选）

**多线程使用示例**（使用64个CPU）：
```bash
python3 scripts/calculate_shapley/calculate_shapley.py \
    --result_dir results/claude-3-haiku-20240307/claude-3-haiku-20240307_42 \
    --n_samples 1000 \
    --n_threads 64 \
    --seed 42
```

### 批量计算

```bash
cd SocialLLM  # or path to SocialLLM directory

# 自动识别所有模型
bash scripts/calculate_shapley/batch_calculate_shapley.sh

# 或者手动指定模型（修改脚本中的 MODEL_NAMES 变量）
# MODEL_NAMES=("claude-3-haiku-20240307" "gpt-4o-mini")
```

**批量脚本特性**：
- 自动识别 `results/` 目录下所有模型
- 自动识别每个模型下所有结果目录（排除 `_rm` 后缀的目录）
- 跳过已计算的结果（如果 `shapley/shapley_attribution_timeseries.csv` 已存在）
- 记录详细的日志信息
- 支持手动指定模型列表

**配置参数**（在脚本中修改）：
- `MODEL_NAMES`: 模型名称列表（空数组表示自动识别所有模型）
- `N_SAMPLES`: 蒙特卡洛采样次数（默认：1000）
- `N_THREADS`: 并行线程数（默认：64，设为 0 则使用 CPU 核心数，建议设置为实际 CPU 核心数）
- `SEED`: 随机数种子（默认：42）

**多线程配置**：
- 脚本使用 `ProcessPoolExecutor` 进行并行计算，支持任意数量的 CPU 核心
- 每个 (agent_id, timestep) 组合作为一个独立任务并行计算
- 建议将 `N_THREADS` 设置为实际可用的 CPU 核心数（如 64）以获得最佳性能

## 输出格式

### CSV 文件

保存为：`{result_dir}/shapley/shapley_attribution_timeseries.csv`

格式：
```csv
agent_id,timestep,shapley_value
0,0,0.0123
0,1,0.0234
1,0,0.0156
...
```

### 统计文件

保存为：`{result_dir}/shapley/shapley_stats.json`

包含：
- `method`: 计算方法（"monte_carlo_shapley"）
- `num_agents`: Agent 数量
- `num_steps`: 模拟步数
- `max_risk_timestep`: 最高风险时间步
- `max_risk`: 最高风险值
- `n_samples`: 采样次数
- `seed`: 随机数种子
- `n_threads`: 并行线程数
- `total_combinations`: 总计算组合数
- `shapley_stats`: 统计信息（mean, std, min, max, sum）

## 算法说明

### Shapley Value 计算公式

对于 agent i 在时间步 t 的 Shapley value：

```
φ_i(t) = Σ_{S ⊆ N\{i}} |S|!(|N|-|S|-1)!/|N|! * [v(S ∪ {i}) - v(S)]
```

其中：
- N 是所有 agent 集合
- S 是 agent 子集（不包含 agent i）
- v(S) 是子集 S 的边际贡献（运行反事实模拟，mask 掉 N\S 的 agent 在时间步 t 的动作）
- 使用蒙特卡洛方法近似计算

### 反事实模拟

对于每个子集 S：
1. Mask 掉所有不在 S 中的 agent 在时间步 t 的动作
2. 运行完整的反事实模拟（从 timestep 0 到 num_steps）
3. 获取目标时间步（max_risk_timestep）的风险值作为 v(S)

### Baseline 策略

当 agent 被 mask 时：
- 该 agent 不发帖
- 该 agent 不互动（不看帖子、不点赞/点踩）
- 该 agent 发的帖子从其他 agent 的视野中移除（其他 agent 看不到该 agent 的帖子）

## 依赖要求

- Python 3.7+
- 依赖包：
  - `pandas`
  - `numpy`
  - `tqdm`
  - `pyyaml`（用于加载配置文件）
- 项目依赖：
  - `counterfactual.py`（反事实模拟模块）
  - `utils.py`（工具函数）

## 注意事项

1. **性能考虑**：
   - 计算 Shapley value 需要运行大量反事实模拟，计算时间较长
   - **强烈建议使用并行计算**（`--n_threads`）加速，可以充分利用多核 CPU（如 64 核）
   - 每个 (agent_id, timestep) 组合作为独立任务并行计算，理论上可以线性加速
   - 可以根据需要调整采样次数（`--n_samples`）平衡精度和速度
   - 使用 64 个 CPU 核心时，计算速度可以提升数十倍

2. **内存使用**：
   - 大量反事实模拟可能占用较多内存
   - 如果内存不足，可以减少并行线程数

3. **结果验证**：
   - 计算完成后，检查 `shapley_stats.json` 中的统计信息
   - 如果 Shapley value 的 sum 接近 0，说明计算可能有问题

4. **配置文件**：
   - 脚本会自动查找 `config/config.yaml` 文件加载配置参数
   - 如果找不到配置文件，会使用默认参数（会输出警告）

## 日志文件

批量计算脚本会在 `logs/` 目录下生成日志文件：
- 文件名格式：`batch_calculate_shapley_YYYYMMDD_HHMMSS.log`
- 包含详细的执行日志、成功/失败统计等信息

## 示例

### 计算单个结果

```bash
python3 scripts/calculate_shapley/calculate_shapley.py \
    --result_dir results/claude-3-haiku-20240307/claude-3-haiku-20240307_42 \
    --n_samples 1000 \
    --n_threads 8
```

### 批量计算所有模型

```bash
bash scripts/calculate_shapley/batch_calculate_shapley.sh
```

### 查看结果

```bash
# 查看 CSV 结果
head results/claude-3-haiku-20240307/claude-3-haiku-20240307_42/shapley/shapley_attribution_timeseries.csv

# 查看统计信息
cat results/claude-3-haiku-20240307/claude-3-haiku-20240307_42/shapley/shapley_stats.json
```

## 多线程性能优化

### 使用64个CPU核心

脚本完全支持多线程并行计算，可以充分利用64个CPU核心：

**单个结果计算**：
```bash
python3 scripts/calculate_shapley/calculate_shapley.py \
    --result_dir results/claude-3-haiku-20240307/claude-3-haiku-20240307_42 \
    --n_samples 1000 \
    --n_threads 64
```

**批量计算**（修改 `batch_calculate_shapley.sh` 中的 `N_THREADS=64`）：
```bash
# 编辑脚本，设置 N_THREADS=64
vim scripts/calculate_shapley/batch_calculate_shapley.sh

# 运行批量计算
bash scripts/calculate_shapley/batch_calculate_shapley.sh
```

### 性能说明

- **并行策略**：每个 (agent_id, timestep) 组合作为独立任务，使用 `ProcessPoolExecutor` 并行计算
- **加速效果**：理论上可以接近线性加速，64个CPU核心可以显著缩短计算时间
- **内存使用**：每个进程会独立加载配置和运行反事实模拟，内存使用会随线程数增加
- **建议配置**：
  - 64个CPU核心：`--n_threads 64`
  - 自动检测：`--n_threads 0`（使用系统CPU核心数）
  - 单线程调试：`--n_threads 1`
