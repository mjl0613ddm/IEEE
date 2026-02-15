# Accuracy Experiment (EconAgent)

验证 Monte Carlo Shapley 近似精度（与 exact Shapley 对比）。

## 脚本

| 脚本 | 用途 |
|------|------|
| `compute_shapley_error.py` | 计算 exact vs MC Shapley 误差 |
| `compute_cosine_similarity.py` | 余弦相似度 |
| `run_counterfactual_analysis.py` | 批量反事实分析（编辑脚本内 configs） |
| `run_shapley_error_example.sh` | 示例：单模型 Shapley 误差 |

## 用法

```bash
# 编辑 run_shapley_error_example.sh 中的 REAL_ACTIONS_JSON, OUTPUT_DIR
bash run_shapley_error_example.sh

# 或直接调用 Python（不同模型改 --real_actions_json 和 --output_dir）
python compute_shapley_error.py \
    --real_actions_json data/{model}-verify/actions_json/all_actions.json \
    --baseline_actions_json data/baseline-verify/actions_json/all_actions.json \
    --num_agents 4 --episode_length 5 \
    --output_dir results/shapley_error_analysis
```
