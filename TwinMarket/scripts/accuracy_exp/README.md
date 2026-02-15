# Accuracy Experiment (TwinMarket)

验证 Monte Carlo Shapley 近似精度（与 exact Shapley 对比）。

## 脚本

| 脚本 | 用途 |
|------|------|
| `analyze_error.py` | 分析 exact vs MC Shapley 误差 |
| `aggregate_shapley_errors.py` | 汇总多模型误差结果 |
| `compute_cosine_similarity.py` | 余弦相似度 |
| `batch_compute_shapley.sh` | 批量计算多模型 Shapley 并分析 |
| `run_shapley_example.sh` | 示例：单模型 Shapley 计算与误差分析 |
| `run_simulations.sh` | 运行模拟 |
| `create_sys_4_db.sh` | 创建 sys_4 数据库 |
| `create_belief_4_csv.sh` | 创建 belief CSV |

## 用法

```bash
# 1. 运行模拟（可选）
bash run_simulations.sh

# 2. 单模型示例：编辑 run_shapley_example.sh 中的 MODEL
bash run_shapley_example.sh

# 3. 批量多模型
bash batch_compute_shapley.sh
```

> 注：`batch_compute_shapley.sh` 和 `run_shapley_example.sh` 依赖 `compute_exact_shapley.py`、`compute_mc_shapley.py`。若缺失，请从论文实验配置或上游仓库补充。
