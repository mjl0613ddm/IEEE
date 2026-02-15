# Risk Feature 计算脚本使用说明

本目录包含用于计算和汇总EconAgent项目风险特征的脚本。

## 文件说明

### 1. `calculate_risk_features.py`
单个模型/seed的风险特征计算脚本。

**功能：**
- 加载Shapley数据和交易记录
- 计算5个风险特征指标
- 保存风险特征JSON文件和5个npy文件

**修改的指标：**
1. **相对风险延迟（L_tm）**：改为计算第一次风险累积到90%的时刻
   - 公式：$L_{\text{tm}} = (T - T^*) / T$
   - 其中 $T^* = \min \{t' \mid \sum_{t=1}^{t'} \phi^{\text{tm}}_{t} > 0.9 \cdot \rho \}$
   
2. **风险-不稳定性相关系数（C_ag）**：先对贡献值取绝对值，再计算相关系数
   - 计算 $|\phi^{\text{ag}}_i|$ 与 $\sigma^{\text{ag}}_i$ 的Pearson相关系数

3. **行为风险集中度（G_be）**：按工作状态和消费比例分8个类型
   - 工作状态：0（不工作）或 1（工作）
   - 消费比例：4个区间（0-25%, 25-50%, 50-75%, 75-100%）
   - 组合：2 × 4 = 8种类型（类型0-7）

**使用方法：**
```bash
python calculate_risk_features.py \
    --data_dir /path/to/datas/{model}/{model}_{seed} \
    [--output_dir /path/to/output/risk_features_exp/{model}/{model}_{seed}]
```

**参数说明：**
- `--data_dir` (必需): 数据目录，包含shapley、action_table、actions_json文件夹
- `--shapley_dir`: Shapley数据目录（默认: {data_dir}/shapley）
- `--action_table_dir`: Action table目录（默认: {data_dir}/action_table）
- `--actions_json_dir`: Actions JSON目录（默认: {data_dir}/actions_json）
- `--output_dir`: 输出目录（默认: {data_dir}/risk_features_exp）

**输出文件：**
- `risk_features.json`: 包含5个指标的JSON文件
- `risk_evolution.npy`: T维向量（每个时间步的风险贡献）
- `behaviour_aggregated.npy`: 8维向量（8种work+consumption类型聚合）
- `attribution_matrix.npy`: N×T矩阵（归因矩阵）
- `time_aggregated.npy`: T维向量（时间聚合）
- `agent_aggregated.npy`: N维向量（Agent聚合）

### 2. `batch_calculate_risk_features.py`
并行批量计算所有模型的risk features。

**功能：**
- 自动发现所有包含shapley结果的模型/seed目录
- 并行执行计算任务
- 记录成功/失败的任务
- 生成执行日志

**使用方法：**
```bash
# 使用默认设置（串行）
python batch_calculate_risk_features.py

# 使用64个并行worker
python batch_calculate_risk_features.py --max_workers 64

# 只处理指定模型（部分匹配）
python batch_calculate_risk_features.py --max_workers 64 --models claude gpt
```

**参数说明：**
- `--datas_root`: Datas根目录（默认: PROJECT_ROOT/datas）
- `--models`: 要处理的模型列表（部分匹配，例如: claude gpt）
- `--max_workers` / `-j`: 最大并行任务数（默认: 1）

**输出：**
- 执行日志保存在 `logs/batch_run_YYYYMMDD_HHMMSS.log`
- 每个模型的结果保存在 `results/risk_features_exp/{model}/{model}_{seed}/`

### 3. `batch_calculate_risk_features.sh`
Shell脚本包装器，可选择使用bash串行执行或调用Python并行脚本。

**使用方法：**
```bash
# 串行执行（使用bash）
bash batch_calculate_risk_features.sh

# 并行执行（自动调用Python脚本）
bash batch_calculate_risk_features.sh --max_workers 64

# 指定模型
bash batch_calculate_risk_features.sh --max_workers 64 --models claude gpt
```

### 4. `aggregate_risk_features.py`
汇总所有模型的risk features计算结果。

**功能：**
- 读取所有模型的所有计算结果
- 按模型汇总（均值、标准差、最小值、最大值、中位数等）
- 生成模型级别的汇总JSON和CSV
- 生成所有模型的对比JSON和CSV

**使用方法：**
```bash
python aggregate_risk_features.py [--risk_feature_root /path/to/risk_features_exp]
```

**参数说明：**
- `--risk_feature_root`: Risk feature根目录（默认: PROJECT_ROOT/results/risk_features_exp）

**输出文件：**
- 单模型汇总：
  - `results/risk_features_exp/{model}/{model}_summary.json`
  - `results/risk_features_exp/{model}/{model}_summary.csv`
- 全模型对比：
  - `results/risk_features_exp/all_models_summary.json`
  - `results/risk_features_exp/all_models_summary.csv`

### 5. `visualize_behavior_risk.py`
行为风险（G_be指标）可视化脚本，生成行为聚类风险分布图。

**功能：**
- 为每个模型生成行为风险分布图（显示所有seed的数据）
- 生成所有模型的对比图
- 可视化8种行为类别的风险贡献（工作状态 × 消费比例区间）

**使用方法：**
```bash
# 可视化所有模型（默认不过滤异常值）
python visualize_behavior_risk.py

# 只可视化指定模型
python visualize_behavior_risk.py --model claude

# 启用异常值过滤
python visualize_behavior_risk.py --filter_outliers

# 启用异常值过滤并自定义阈值
python visualize_behavior_risk.py --filter_outliers --strict_threshold 1.8 --normal_threshold 2.0

# 指定risk feature根目录
python visualize_behavior_risk.py --risk_feature_root /path/to/risk_features_exp
```

**参数说明：**
- `--risk_feature_root`: Risk feature根目录（默认: PROJECT_ROOT/results/risk_features_exp）
- `--model`: 只可视化指定模型（部分匹配，默认: 所有模型）
- `--filter_outliers`: 启用异常值过滤（默认: False，不过滤）
- `--strict_threshold`: 严格类别的Z-score阈值（类别2和6，默认: 1.8）
- `--normal_threshold`: 普通类别的Z-score阈值（默认: 2.0）

**输出文件：**
- 单模型图：
  - `results/risk_features_exp/{model}/behavior_risk_distribution.png`
- 全模型对比图：
  - `results/risk_features_exp/visualizations/behavior_risk/all_models_behavior_comparison.png`

**可视化内容：**
- 柱状图显示每个行为类别的平均风险贡献（带误差棒）
- 散点图叠加显示所有seed的个体数据点
- 正负值用不同颜色区分（红色=正贡献，蓝色=负贡献）
- 包含统计信息（seed数量、G_be均值等）

**归一化逻辑：**
1. **单模型可视化**：
   - 每个seed的 `behaviour_aggregated.npy` 先按自身的**有符号总和**归一化
   - **基础过滤（默认启用）**：
     - 总和若接近0（<1e-10）则跳过该seed
     - 归一化后如果任何一个类别的值超出合理范围（>1.0 或 <-1.0），过滤掉该seed
     - 这样可以避免总和过小导致归一化后值被异常放大的问题
   - **异常值过滤（可选，默认关闭）**：
     - 使用 `--filter_outliers` 参数启用
     - 多层异常值检测：
       - 第一层：如果归一化后绝对值最大值超过5，直接过滤
       - 第二层：对每个类别单独使用迭代Z-score方法检测
         - 对于类别2（work=0, cons=50-75%）和类别6（work=1, cons=50-75%），使用更严格的阈值（默认1.8，可通过`--strict_threshold`调整）
         - 对于其他类别，使用普通阈值（默认2.0，可通过`--normal_threshold`调整）
         - 迭代检测：每次过滤后重新计算统计量，直到没有新的异常值
     - 注意：如果过滤后某些模型在特定类别上仍明显高于其他模型，可能是模型间的系统性差异而非异常值
   - 然后在该模型的所有seed上计算均值、标准差等统计量
   
2. **多模型对比**：
   - 每个模型先计算其所有seed的均值（已归一化的数据，异常值已过滤）
   - 然后进行模型间对比，确保每个模型有相等的权重
   - 不会因为不同模型seed数量不同而影响对比结果（例如：llama有11个seed，qwen有4个seed，但它们在对比时权重相等）
   - 图例中会显示每个模型的seed数量（n=...），便于了解统计可靠性

## 5个风险特征指标（按顺序：L_tm, G_ag, C_ag, Z_ag, G_be）

| 指标名称 | 变量名 | 数学符号 | 说明 |
|---------|--------|---------|------|
| 相对风险延迟 | L_tm | L_tm | 90%风险累积时刻的相对延迟（**已修改**） |
| 风险个体稀疏性 | G_ag | G_ag | Agent风险集中度（基尼系数） |
| 风险-不稳定性相关系数 | C_ag | C_ag | Agent贡献绝对值与不稳定性的相关系数（**已修改**） |
| 风险个体协同性 | Z_ag | Z_ag | Agent风险同步性 |
| 行为风险集中度 | G_be | G_be | 行为模式风险集中度（基尼系数），按工作状态和消费比例分8个类型（工作/不工作 × 4个消费区间） |

### G_be指标分类（8种类型）

- **类型0**: 不工作(0) + 消费0-25% (0.0 - 0.25)
- **类型1**: 不工作(0) + 消费25-50% (0.25 - 0.50)
- **类型2**: 不工作(0) + 消费50-75% (0.50 - 0.75)
- **类型3**: 不工作(0) + 消费75-100% (0.75 - 1.0)
- **类型4**: 工作(1) + 消费0-25% (0.0 - 0.25)
- **类型5**: 工作(1) + 消费25-50% (0.25 - 0.50)
- **类型6**: 工作(1) + 消费50-75% (0.50 - 0.75)
- **类型7**: 工作(1) + 消费75-100% (0.75 - 1.0)

**计算公式**：`category = work_status * 4 + consumption_category`

## 完整工作流程

### 步骤1：批量计算所有模型的risk features
```bash
cd /mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/scripts/risk_features_exp

# 使用64个并行worker计算所有模型
python batch_calculate_risk_features.py --max_workers 64

# 或只计算指定模型
python batch_calculate_risk_features.py --max_workers 64 --models claude gpt
```

### 步骤2：汇总所有结果
```bash
# 汇总所有模型的risk features
python aggregate_risk_features.py

# 结果将保存在：
# - results/risk_features_exp/{model}/{model}_summary.json/csv（单模型汇总）
# - results/risk_features_exp/all_models_summary.json/csv（全模型对比）
```

### 步骤3：可视化行为风险分布
```bash
# 生成所有模型的行为风险可视化图
python visualize_behavior_risk.py

# 或只生成指定模型的可视化图
python visualize_behavior_risk.py --model claude

# 结果将保存在：
# - results/risk_features_exp/{model}/behavior_risk_distribution.png（单模型图）
# - results/risk_features_exp/visualizations/behavior_risk/all_models_behavior_comparison.png（对比图）
```

### 步骤4：查看结果
```bash
# 查看单模型汇总
cat results/risk_features_exp/claude/claude_summary.csv

# 查看全模型对比
cat results/risk_features_exp/all_models_summary.csv

# 查看可视化图片
ls results/risk_features_exp/claude/behavior_risk_distribution.png
ls results/risk_features_exp/visualizations/behavior_risk/all_models_behavior_comparison.png
```

## 目录结构

```
results/risk_features_exp/
├── {model}/
│   ├── {model}_{seed}/
│   │   ├── risk_features.json          # 单个结果
│   │   ├── risk_evolution.npy
│   │   ├── behaviour_aggregated.npy    # 8维向量
│   │   ├── attribution_matrix.npy
│   │   ├── time_aggregated.npy
│   │   └── agent_aggregated.npy
│   ├── {model}_summary.json            # 模型汇总
│   ├── {model}_summary.csv
│   └── behavior_risk_distribution.png  # 单模型可视化图
├── visualizations/
│   └── behavior_risk/
│       └── all_models_behavior_comparison.png  # 全模型对比图
├── all_models_summary.json              # 全模型对比
└── all_models_summary.csv
```

## 数据格式说明

### EconAgent数据格式

- **Shapley文件**：`shapley/shapley_values.npy` (N×T矩阵，N=10个agents，T=50个timesteps)
- **Stats文件**：`shapley/shapley_stats.json` (包含num_agents, episode_length, seed等信息)
- **Action table文件**：`action_table/action_table.csv` (包含agent_id, timestep, endogenous_Consumption Rate等列)
- **Actions JSON文件**：`actions_json/all_actions.json` (包含work状态，格式：`{"step_1": {"0": [1, 25.0], ...}}`)

### 与TwinMarket的差异

| 方面 | TwinMarket | EconAgent |
|------|-----------|-----------|
| Shapley文件 | `shapley_matrix_*.npy` | `shapley_values.npy` |
| 数据路径 | `results/{model}/{model}_{seed}/` | `datas/{model}/{model}_{seed}/` |
| 时间维度 | 日期列表 | timesteps (1-50) |
| Agent ID | 字符串user_ids | 数字 (0-9) |
| 行为数据 | `simulation_results/transactions.csv` | `action_table/action_table.csv` + `actions_json/all_actions.json` |
| G_be分类 | 10种股票类型 | 8种类型（工作/不工作 × 4个consumption区间） |

## 注意事项

1. **数据要求：**
   - 每个模型/seed目录下必须有 `shapley/` 文件夹
   - `shapley/` 文件夹中需要包含 `shapley_values.npy` 和 `shapley_stats.json` 文件
   - 可选：`action_table/action_table.csv` 和 `actions_json/all_actions.json`（用于G_be指标计算）

2. **并行执行：**
   - 使用 `--max_workers` 参数控制并行数量
   - 建议根据CPU核心数设置（例如64核CPU可以使用32-64个workers）
   - 每个任务会占用一定的内存，注意不要设置过多workers导致内存不足

3. **错误处理：**
   - 如果某个模型/seed的计算失败，会记录错误信息并继续处理其他任务
   - 详细错误信息保存在日志文件中
   - 汇总脚本会自动跳过缺失的结果文件

4. **指标说明：**
   - 所有指标的值都在合理范围内（例如基尼系数在[0,1]，相关系数在[-1,1]）
   - 如果某个指标计算失败或数据不足，会返回0或NaN

5. **G_be指标计算：**
   - work状态从 `actions_json/all_actions.json` 读取（每个agent每个timestep的action格式为`[work, consumption]`，work为0或1）
   - consumption rate从 `action_table/action_table.csv` 的 `endogenous_Consumption Rate` 列读取（0-1之间的值）
   - 如果缺少actions数据，G_be指标可能无法正确计算

## 故障排除

### 问题1：找不到shapley文件
**原因：** shapley文件不存在或路径不正确
**解决：** 检查 `{data_dir}/shapley/shapley_values.npy` 是否存在

### 问题2：actions数据加载失败
**原因：** action_table或actions_json目录不存在或文件缺失
**解决：** 这是可选的，如果缺失只会影响G_be指标的计算，其他指标不受影响

### 问题3：并行执行时内存不足
**原因：** workers数量过多导致内存不足
**解决：** 减少 `--max_workers` 参数的值

### 问题4：汇总时找不到结果文件
**原因：** 批量计算未完成或输出路径不正确
**解决：** 检查 `results/risk_features_exp/` 目录下是否有相应的结果文件

## 示例

### 示例1：计算单个模型
```bash
python calculate_risk_features.py \
    --data_dir /mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/datas/claude/claude_42 \
    --output_dir /mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/results/risk_features_exp/claude/claude_42
```

### 示例2：批量计算所有模型
```bash
cd /mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/scripts/risk_features_exp
python batch_calculate_risk_features.py --max_workers 32
```

### 示例3：只计算指定模型并汇总
```bash
# 计算
python batch_calculate_risk_features.py --max_workers 32 --models claude gpt

# 汇总
python aggregate_risk_features.py
```

### 示例4：生成行为风险可视化图
```bash
# 生成所有模型的可视化图
python visualize_behavior_risk.py

# 只生成claude模型的可视化图
python visualize_behavior_risk.py --model claude
```

## 更新日志

### 2024-01-XX
- 修改时间指标（L_tm）：改为计算90%风险累积时刻
- 修改风险-不稳定性相关系数（C_ag）：先取绝对值再计算
- 修改行为风险集中度（G_be）：按工作状态和消费比例分8个类型（工作/不工作 × 4个消费区间）
- 创建批量计算和汇总脚本
- 适配EconAgent数据格式（shapley_values.npy，timesteps，agent IDs等）
- 添加行为风险可视化脚本（visualize_behavior_risk.py）