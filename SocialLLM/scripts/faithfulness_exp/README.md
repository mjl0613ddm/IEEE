# Faithfulness (SocialLLM)

计算 baseline 归因分数与 faithfulness 指标。

## 快速开始

```bash
# 1. 提取 action table
python extract_action_table.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42

# 2. 批量计算 baselines
bash batch_compute_baselines.sh

# 3. 计算 faithfulness
python compute_faithfulness.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42 --method shapley --metric_type deletion_top_5
```

## 脚本

| 脚本 | 用途 |
|------|------|
| `extract_action_table.py` | 提取 action table |
| `compute_random_baseline.py` | Random |
| `compute_loo_baseline.py` | LOO |
| `compute_llm_baseline.py` | LLM (需 API) |
| `compute_mast_baseline.py` | MAST (需 API) |
| `batch_compute_baselines.sh` | 批量 baselines |
| `compute_faithfulness.py` | Faithfulness |

## 输出

- Baselines: `results/{model}/{seed}/faithfulness_exp/{method}/`
- 汇总: `results/faithfulness_exp/{model}/faithfulness_summary.csv`
