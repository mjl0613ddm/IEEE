# SocialLLM Faithfulness Experiment Scripts

本目录包含用于计算faithfulness实验baseline方法的脚本。这些脚本为每个(agent_id, timestep)对生成重要性分数，用于与Shapley value进行对比分析。

## 目录结构

```
scripts/faithfulness_exp/
├── extract_action_table.py      # 提取action table
├── compute_random_baseline.py   # Random baseline方法
├── compute_loo_baseline.py      # Leave-One-Out baseline方法
├── compute_llm_baseline.py      # LLM as a Judge baseline方法
├── compute_mast_baseline.py    # MAST baseline方法
├── batch_compute_baselines.sh   # 批量运行脚本
└── README.md                    # 本文档
```

## 方法说明

### 1. Random Baseline
为所有(agent_id, timestep)对随机生成0-1之间的分数。

**输出格式**: `faithfulness_exp/random/random_attribution_timeseries.csv`
- 列: `agent_id`, `timestep`, `random_value`

### 2. Leave-One-Out (LOO) Baseline
通过移除每个action并计算风险下降值来评估action的重要性。

**输出格式**: `faithfulness_exp/loo/loo_attribution_timeseries.csv`
- 列: `agent_id`, `timestep`, `loo_value`
- 值表示移除该action后风险下降的幅度（风险下降越多，说明该action越重要）

### 3. LLM as a Judge Baseline
使用LLM分析社交媒体模拟轨迹，为每个agent-timestep对分配风险归因分数。

**输出格式**: `faithfulness_exp/llm/llm_attribution_timeseries.csv`
- 列: `agent_id`, `timestep`, `llm_value`
- 值范围: [0, 1]

**依赖**: 
- 需要先运行`extract_action_table.py`生成action table
- 需要配置LLM API（见下方配置说明）

### 4. MAST Baseline
基于LLM-as-a-Judge，使用MAST定义和示例来指导LLM评分。

**输出格式**: `faithfulness_exp/mast/mast_attribution_timeseries.csv`
- 列: `agent_id`, `timestep`, `mast_value`
- 值范围: [0, 1]

**依赖**: 
- 需要先运行`extract_action_table.py`生成action table
- 需要配置LLM API（见下方配置说明）

## 使用方法

### 1. 提取Action Table

首先需要提取action table（LLM和MAST方法需要）：

#### 单个结果目录提取
```bash
python scripts/faithfulness_exp/extract_action_table.py \
    --result_dir results/gpt-4o-mini/gpt-4o-mini_42
```

#### 批量提取（自动跳过_rm文件夹）
```bash
bash scripts/faithfulness_exp/batch_extract_action_table.sh
```

**输出**: `results/gpt-4o-mini/gpt-4o-mini_42/action_table/action_table.csv`

**注意**: 批量脚本会自动跳过名称以`_rm`结尾的文件夹

**列说明**:
- `agent_id`: Agent ID
- `timestep`: 时间步（0-indexed）
- `posted`: 是否发帖（1=发帖，0=未发帖）
- `view_count`: 浏览帖子数量
- `like_count`: 点赞数量
- `dislike_count`: 点踩数量
- `belief`: 该agent在该timestep的belief值（从-1到+1）

### 2. 运行单个Baseline方法

#### Random Baseline
```bash
python scripts/faithfulness_exp/compute_random_baseline.py \
    --result_dir results/gpt-4o-mini/gpt-4o-mini_42 \
    --seed 42
```

#### LOO Baseline
```bash
python scripts/faithfulness_exp/compute_loo_baseline.py \
    --result_dir results/gpt-4o-mini/gpt-4o-mini_42
```

#### LLM Baseline
```bash
python scripts/faithfulness_exp/compute_llm_baseline.py \
    --result_dir results/gpt-4o-mini/gpt-4o-mini_42 \
    --config config/api.yaml \
    --rows-per-batch 100 \
    --max-workers 10
```

#### MAST Baseline
```bash
python scripts/faithfulness_exp/compute_mast_baseline.py \
    --result_dir results/gpt-4o-mini/gpt-4o-mini_42 \
    --config config/api.yaml \
    --rows-per-batch 100 \
    --max-workers 10
```

**参数说明**:
- `--rows-per-batch`: 每批处理的行数（默认：100），用于避免token上限
- `--max-workers`: 并发线程数（默认：1表示串行）。API调用是I/O密集型，可以设置较大值（如10-20）来加速

### 3. 批量运行所有Baseline方法

使用批量脚本自动处理所有结果目录：

```bash
bash scripts/faithfulness_exp/batch_compute_baselines.sh
```

**配置**（编辑脚本中的变量）:
- `MODEL_NAMES`: 要处理的模型列表（空数组表示自动识别所有模型）
- `BASELINE_METHODS`: 要运行的方法列表（默认: random loo llm mast）
- `LLM_CONFIG_FILE`: LLM配置文件路径（默认: `config/api.yaml`）
  - 如果配置文件不存在，脚本会使用环境变量 `OPENAI_API_KEY`、`OPENAI_MODEL`、`OPENAI_BASE_URL`
  - 也可以设置为具体的配置文件，如: `config/api_gpt-4o-mini.yaml`
- `LLM_ROWS_PER_BATCH`: 每批处理的行数（默认: 100），用于避免token上限
- `LLM_MAX_WORKERS`: 并发线程数（默认: 10），用于加速LLM/MAST方法的API调用

**API配置方式**（二选一）:
1. **使用配置文件**: 创建或修改 `config/api.yaml`，包含 `api_key`、`model_name`、`base_url` 等字段
2. **使用环境变量**: 在运行脚本前设置环境变量
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_MODEL="gpt-4o"
   export OPENAI_BASE_URL="https://api.openai.com/v1"
   ```

**特性**:
- 自动跳过`_rm`结尾的文件夹
- 自动提取action table（如果需要）
- 跳过已存在的结果文件
- 详细的日志记录

## 输出格式

所有baseline方法的输出格式与Shapley value一致：

**CSV文件**: `{method}_attribution_timeseries.csv`
```csv
agent_id,timestep,{method}_value
0,0,0.123456
0,1,0.234567
...
```

**统计文件**: `{method}_stats.json`
```json
{
  "method": "random|loo|llm|mast",
  "num_agents": 20,
  "max_risk_timestep": 15,
  "max_risk": 0.607805,
  "initial_risk": 0.080390,
  "score_stats": {
    "mean": 0.5,
    "std": 0.288675,
    "min": 0.0,
    "max": 1.0,
    "sum": 150.0,
    "median": 0.5
  }
}
```

## LLM配置

LLM和MAST方法需要配置LLM API。配置文件格式（YAML或JSON）：

**config/api.yaml**:
```yaml
api_key: "your-api-key"
model: "gpt-4o"
base_url: "https://api.openai.com/v1"
temperature: 0.7
timeout: 300
```

**环境变量**（可选）:
- `OPENAI_API_KEY`: API密钥
- `OPENAI_MODEL`: 模型名称（默认: gpt-4o）
- `OPENAI_BASE_URL`: API基础URL（默认: https://api.openai.com/v1）
- `OPENAI_TIMEOUT`: 超时时间（秒，默认: 300）

## 性能优化

1. **LOO方法**: 需要运行大量counterfactual simulation，计算时间较长。建议在计算资源充足时运行。

2. **LLM/MAST方法**: 
   - **切片输入**: 使用`--rows-per-batch`参数控制每批处理的行数（默认: 100），避免token达到上限
     - 较大的batch size可以减少API调用次数，但可能增加token消耗和超时风险
     - 较小的batch size更稳定，但API调用次数更多
   - **并发处理**: 使用`--max-workers`参数控制并发线程数（默认: 1表示串行）
     - API调用是I/O密集型操作，与CPU无关，主要受网络带宽和API限流影响
     - 可以设置较大的并发数（如10-20）来加速，但需要注意API的速率限制
     - 示例：`--max-workers 10` 表示同时处理10个batch

3. **批量运行**: 使用`batch_compute_baselines.sh`可以并行处理多个结果目录（如果系统资源允许）。

## 注意事项

1. **Action Table依赖**: LLM和MAST方法需要先运行`extract_action_table.py`生成action table。

2. **结果目录结构**: 确保结果目录包含以下文件：
   - `results.json`: 包含模拟结果和风险指标
   - `actions.json`: 包含所有agent的动作历史（用于提取action table）

3. **_rm文件夹**: 批量脚本会自动跳过名称以`_rm`结尾的文件夹（这些是经过筛选后不符合标准的结果）。

4. **已存在结果**: 默认情况下，如果结果文件已存在，脚本会跳过。使用`--skip-existing`可以显式启用此行为。

## 故障排除

1. **LLM API错误**: 
   - 检查API密钥是否正确
   - 检查网络连接
   - 尝试减小`--rows-per-batch`参数

2. **LOO计算慢**: 
   - LOO方法需要运行大量simulation，这是正常的
   - 可以考虑减少处理的timestep范围（修改代码）

3. **Action Table缺失**: 
   - 确保已运行`extract_action_table.py`
   - 检查`action_table/action_table.csv`文件是否存在

## 参考

- Shapley value计算: `scripts/calculate_shapley/`
- 结果筛选: `scripts/filter/`
- 风险可视化: `scripts/plot/`
