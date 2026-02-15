# Risk Feature 计算脚本使用说明

本目录包含用于计算和汇总TwinMarket项目风险特征的脚本。

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

**使用方法：**
```bash
python calculate_risk_features.py \
    --results_dir /path/to/results/{model}/{model}_{seed} \
    [--output_dir /path/to/output/risk_feature/{model}/{model}_{seed}]
```

**参数说明：**
- `--results_dir` (必需): 结果根目录，包含shapley和simulation_results文件夹
- `--shapley_dir`: Shapley数据目录（默认: {results_dir}/shapley）
- `--simulation_results_dir`: 模拟结果目录（默认: {results_dir}/simulation_results）
- `--output_dir`: 输出目录（默认: {results_dir}/risk_feature）

**输出文件：**
- `risk_features.json`: 包含5个指标的JSON文件
- `risk_evolution.npy`: T维向量（每个时间步的风险贡献）
- `behaviour_aggregated.npy`: K维向量（K=10，股票类别聚合）
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
python batch_calculate_risk_features.py --max_workers 64 --models deepseek claude
```

**参数说明：**
- `--results_root`: Results根目录（默认: PROJECT_ROOT/results）
- `--models`: 要处理的模型列表（部分匹配，例如: deepseek claude）
- `--max_workers` / `-j`: 最大并行任务数（默认: 1）

**输出：**
- 执行日志保存在 `logs/batch_run_YYYYMMDD_HHMMSS.log`
- 每个模型的结果保存在 `results/risk_feature/{model}/{model}_{seed}/`

### 3. `batch_calculate_risk_features.sh`
Shell脚本包装器，可选择使用bash串行执行或调用Python并行脚本。

**使用方法：**
```bash
# 串行执行（使用bash）
bash batch_calculate_risk_features.sh

# 并行执行（自动调用Python脚本）
bash batch_calculate_risk_features.sh --max_workers 64

# 指定模型
bash batch_calculate_risk_features.sh --max_workers 64 --models deepseek claude
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
python aggregate_risk_features.py [--risk_feature_root /path/to/risk_feature]
```

**参数说明：**
- `--risk_feature_root`: Risk feature根目录（默认: PROJECT_ROOT/results/risk_feature）

**输出文件：**
- 单模型汇总：
  - `results/risk_feature/{model}/{model}_summary.json`
  - `results/risk_feature/{model}/{model}_summary.csv`
- 全模型对比：
  - `results/risk_feature/all_models_summary.json`
  - `results/risk_feature/all_models_summary.csv`

### 5. `visualize_risk_features.py`
Risk Feature可视化脚本。

**功能：**
- L_tm（相对风险延迟）指标的可视化：
  - 单个seed的风险累积过程图（显示从第0步开始的累积曲线和阈值线）
  - 单模型L_tm分布图（显示所有seed的数据点）
  - 所有模型L_tm对比图（汇总对比）
- 动作聚类风险（G_be指标，按股票类别）的可视化：
  - 单模型动作聚类风险分布图（显示所有seed的数据点）
  - 所有模型动作聚类风险对比图（可选）

**使用方法：**
```bash
# 可视化所有模型
python visualize_risk_features.py

# 只可视化指定模型
python visualize_risk_features.py --model deepseek-v3.2

# 跳过单个seed的风险累积过程可视化（如果太慢）
python visualize_risk_features.py --skip_accumulation

# 指定risk_feature根目录
python visualize_risk_features.py --risk_feature_root /path/to/risk_feature
```

**参数说明：**
- `--risk_feature_root`: Risk feature根目录（默认: PROJECT_ROOT/results/risk_feature）
- `--model`: 只可视化指定模型（部分匹配，例如: deepseek-v3.2）
- `--skip_accumulation`: 跳过单个seed的风险累积过程可视化（可加快速度）

**输出文件：**
- 单模型L_tm分布：`results/risk_feature/{model}/L_tm_distribution.png`
- 单模型动作聚类风险分布：`results/risk_feature/{model}/behavior_risk_distribution.png`
- 单个seed风险累积过程图：`results/risk_feature/{model}/{model}_{seed}/risk_accumulation.png`
- 所有模型L_tm对比：`results/risk_feature/visualizations/L_tm/all_models_L_tm_comparison.png`
- 所有模型动作聚类风险对比：`results/risk_feature/visualizations/behavior_risk/all_models_behavior_comparison.png`

**可视化说明：**
- **单个seed风险累积过程图**：显示从第0步开始到T步的风险累积曲线，包含阈值线（90% × ρ）和T*时刻标记
- **单模型L_tm分布图**：显示该模型所有seed的L_tm值（散点图），包含均值、标准差、中位数、最小值、最大值等统计信息
- **单模型动作聚类风险分布图**：显示10种股票的风险贡献（柱状图），包含所有seed的数据点和均值±标准差
- **汇总对比图**：显示所有模型的指标对比，便于模型间比较

## 5个风险特征指标（按顺序：L_tm, G_ag, C_ag, Z_ag, G_be）

| 指标名称 | 变量名 | 数学符号 | 说明 |
|---------|--------|---------|------|
| 相对风险延迟 | L_tm | L_tm | 90%风险累积时刻的相对延迟（**已修改**） |
| 风险个体稀疏性 | G_ag | G_ag | Agent风险集中度（基尼系数） |
| 风险-不稳定性相关系数 | C_ag | C_ag | Agent贡献绝对值与不稳定性的相关系数（**已修改**） |
| 风险个体协同性 | Z_ag | Z_ag | Agent风险同步性 |
| 行为风险集中度 | G_be | G_be | 行为模式风险集中度（基尼系数），同时考虑买入和卖出交易，按股票种类分类（K=10） |

## 完整工作流程

### 步骤1：批量计算所有模型的risk features
```bash
cd /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/scripts/risk_feature

# 使用64个并行worker计算所有模型
python batch_calculate_risk_features.py --max_workers 64

# 或只计算指定模型
python batch_calculate_risk_features.py --max_workers 64 --models deepseek-v3.2 claude-3-haiku
```

### 步骤2：汇总所有结果
```bash
# 汇总所有模型的risk features
python aggregate_risk_features.py

# 结果将保存在：
# - results/risk_feature/{model}/{model}_summary.json/csv（单模型汇总）
# - results/risk_feature/all_models_summary.json/csv（全模型对比）
```

### 步骤3：汇总所有结果
```bash
# 汇总所有模型的risk features
python aggregate_risk_features.py

# 结果将保存在：
# - results/risk_feature/{model}/{model}_summary.json/csv（单模型汇总）
# - results/risk_feature/all_models_summary.json/csv（全模型对比）
```

### 步骤4：生成可视化
```bash
# 生成所有模型的可视化
python visualize_risk_features.py

# 或只可视化指定模型
python visualize_risk_features.py --model deepseek-v3.2

# 如果风险累积过程图生成太慢，可以跳过
python visualize_risk_features.py --skip_accumulation
```

### 步骤5：查看结果
```bash
# 查看单模型汇总
cat results/risk_feature/deepseek-v3.2/deepseek-v3.2_summary.csv

# 查看全模型对比
cat results/risk_feature/all_models_summary.csv

# 查看可视化图片
# - 单模型L_tm分布：results/risk_feature/{model}/L_tm_distribution.png
# - 单模型动作聚类风险：results/risk_feature/{model}/behavior_risk_distribution.png
# - 单个seed风险累积过程：results/risk_feature/{model}/{model}_{seed}/risk_accumulation.png
# - 所有模型L_tm对比：results/risk_feature/visualizations/L_tm/all_models_L_tm_comparison.png
```

## 目录结构

```
results/risk_feature/
├── {model}/
│   ├── {model}_{seed}/
│   │   ├── risk_features.json          # 单个结果
│   │   ├── risk_evolution.npy
│   │   ├── behaviour_aggregated.npy
│   │   ├── attribution_matrix.npy
│   │   ├── time_aggregated.npy
│   │   ├── agent_aggregated.npy
│   │   └── risk_accumulation.png       # 单个seed风险累积过程图（可视化）
│   ├── L_tm_distribution.png            # 单模型L_tm分布图（可视化）
│   ├── behavior_risk_distribution.png   # 单模型动作聚类风险分布图（可视化）
│   ├── {model}_summary.json            # 模型汇总
│   └── {model}_summary.csv
├── visualizations/
│   ├── L_tm/
│   │   └── all_models_L_tm_comparison.png  # 所有模型L_tm对比图
│   └── behavior_risk/
│       └── all_models_behavior_comparison.png  # 所有模型动作聚类风险对比图
├── all_models_summary.json              # 全模型对比
└── all_models_summary.csv
```

## 注意事项

1. **数据要求：**
   - 每个模型/seed目录下必须有 `shapley/` 文件夹
   - `shapley/` 文件夹中需要包含 `shapley_matrix_*.npy` 文件
   - 可选：`shapley_labels_*.npy` 和 `shapley_stats_*.json` 文件
   - 可选：`simulation_results/` 文件夹（用于行为分类）

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

## 故障排除

### 问题1：找不到shapley文件
**原因：** shapley文件不存在或路径不正确
**解决：** 检查 `{results_dir}/shapley/shapley_matrix_*.npy` 是否存在

### 问题2：交易记录加载失败
**原因：** simulation_results目录不存在或交易记录文件缺失
**解决：** 这是可选的，如果缺失只会影响G_be指标的计算，其他指标不受影响

### 问题3：并行执行时内存不足
**原因：** workers数量过多导致内存不足
**解决：** 减少 `--max_workers` 参数的值

### 问题4：汇总时找不到结果文件
**原因：** 批量计算未完成或输出路径不正确
**解决：** 检查 `results/risk_feature/` 目录下是否有相应的结果文件

### 问题5：可视化时找不到matplotlib或seaborn
**原因：** Python环境中没有安装matplotlib或seaborn
**解决：** 安装依赖：`pip install matplotlib seaborn numpy`

## 示例

### 示例1：计算单个模型
```bash
python calculate_risk_features.py \
    --results_dir /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/results/deepseek-v3.2/deepseek-v3.2_42 \
    --output_dir /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/results/risk_feature/deepseek-v3.2/deepseek-v3.2_42
```

### 示例2：批量计算所有模型
```bash
cd /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/scripts/risk_feature
python batch_calculate_risk_features.py --max_workers 32
```

### 示例3：只计算指定模型并汇总
```bash
# 计算
python batch_calculate_risk_features.py --max_workers 32 --models deepseek claude

# 汇总
python aggregate_risk_features.py
```

### 示例4：生成可视化
```bash
# 生成所有模型的可视化
python visualize_risk_features.py

# 只可视化指定模型
python visualize_risk_features.py --model deepseek-v3.2

# 快速模式（跳过单个seed的风险累积过程图）
python visualize_risk_features.py --skip_accumulation
```

## 更新日志

### 2024-01-XX
- 修改时间指标（L_tm）：改为计算90%风险累积时刻
- 修改风险-不稳定性相关系数（C_ag）：先取绝对值再计算
- 修改行为风险集中度（G_be）：同时考虑买入和卖出交易
- 创建批量计算和汇总脚本
- 创建可视化脚本：
  - 单个seed的风险累积过程可视化
  - 单模型L_tm分布可视化（显示所有seed数据点）
  - 所有模型L_tm对比可视化
  - 单模型动作聚类风险可视化（显示所有seed数据点）
  - 所有模型动作聚类风险对比可视化