# Faithfulness实验脚本

## 概述

本目录包含faithfulness实验的相关脚本，用于计算和评估不同归因方法的准确性。

## 脚本说明

### Baseline方法计算脚本

用于生成baseline方法的分数矩阵，包括：
- `extract_action_table.py`: 提取action table数据
- `compute_random_baseline.py`: 生成随机分数
- `compute_loo_baseline.py`: 计算LOO baseline分数
- `compute_llm_baseline.py`: 计算LLM baseline分数
- `compute_mast_baseline.py`: 计算MAST baseline分数
- `batch_compute_baselines.py` / `batch_compute_baselines.sh`: 批量运行所有baseline方法

### Faithfulness计算脚本

用于计算faithfulness指标，评估归因方法的准确性：
- `compute_faithfulness.py`: 计算单个模型的faithfulness指标
- `batch_compute_faithfulness.sh`: 批量运行所有模型的faithfulness计算
- `generate_faithfulness_summary.py`: 生成汇总CSV文件

## 使用方法

### 运行Faithfulness实验

```bash
# 计算单个指标
python scripts/faithfulness_exp/compute_faithfulness.py gpt/gpt_42 --method shapley --metric_type deletion_top_5

# 批量运行所有模型
./scripts/faithfulness_exp/batch_compute_faithfulness.sh

# 生成汇总文件
python scripts/faithfulness_exp/generate_faithfulness_summary.py gpt
```

**参数说明**:
- `--method`: 归因方法 (`shapley`, `random`, `llm`, `mast`, `loo`)
- `--metric_type`: 指标类型 (`deletion_top_5`, `deletion_top_10`, `insertion_top_5`, `insertion_top_10`)

### 运行Baseline方法计算

```bash
# 批量运行所有baseline方法
python scripts/faithfulness_exp/batch_compute_baselines.py

# 或使用shell脚本
./scripts/faithfulness_exp/batch_compute_baselines.sh
```

## 输出路径

### Baseline方法输出

- 分数矩阵: `datas/{model}/{model_id}/faithfulness_exp/{method}/{method}_scores.npy`
- 统计信息: `datas/{model}/{model_id}/faithfulness_exp/{method}/{method}_stats.json`

### Faithfulness结果输出

- 单个结果: `results/faithfulness_exp/{model}/{model_id}/faithfulness_results_{method}_{metric_type}.json`
- 汇总文件: `results/faithfulness_exp/{model}/faithfulness_summary.csv`

## 配置

LLM和MAST方法需要API配置，推荐在统一配置文件中设置：

`scripts/faithfulness_exp/config.yaml`:
```yaml
api_key: your_api_key_here
model: gpt-4o
base_url: http://your-api-base-url/v1
temperature: 0.7
timeout: 300
```

## 日志文件

批量运行的日志保存在: `scripts/faithfulness_exp/logs/`
