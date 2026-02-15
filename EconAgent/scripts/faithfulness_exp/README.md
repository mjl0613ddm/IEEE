# Faithfulness 实验

评估不同归因方法的 faithfulness。

## 脚本

| 脚本 | 用途 |
|------|------|
| `extract_action_table.py` | 提取 action table |
| `compute_random_baseline.py` | 随机 baseline |
| `compute_loo_baseline.py` | LOO baseline |
| `compute_llm_baseline.py` | LLM baseline |
| `compute_mast_baseline.py` | MAST baseline |
| `batch_compute_baselines.py` | 批量计算 baselines |
| `compute_faithfulness.py` | 计算 faithfulness |
| `batch_compute_faithfulness.sh` | 批量计算 |
| `generate_faithfulness_summary.py` | 生成汇总 CSV |

## 快速开始

```bash
# 1. 批量计算 baselines
python scripts/faithfulness_exp/batch_compute_baselines.py

# 2. 批量计算 faithfulness
./scripts/faithfulness_exp/batch_compute_faithfulness.sh

# 3. 生成汇总
python scripts/faithfulness_exp/generate_faithfulness_summary.py gpt
```

**单次运行**：
```bash
python scripts/faithfulness_exp/compute_faithfulness.py gpt/gpt_42 --method shapley --metric_type deletion_top_5
```

## 配置

LLM/MAST 需配置 `scripts/faithfulness_exp/config.yaml`：

```yaml
api_key: your_api_key
model: gpt-4o
base_url: http://your-api/v1
```

## 输出

- Baselines: `datas/{model}/{model_id}/faithfulness_exp/{method}/`
- 结果: `results/faithfulness_exp/{model}/`
- 日志: `scripts/faithfulness_exp/logs/`
