# Faithfulness (TwinMarket)

评估归因方法的 faithfulness（deletion/insertion 曲线）。

## 快速开始

```bash
# 1. 提取 action 特征
python extract_action_features.py [--models gpt-4o-mini]

# 2. 计算 faithfulness
python compute_faithfulness.py --log_dir results/gpt-4o-mini/gpt-4o-mini_42 --method shapley --max_actions 10

# 3. 批量计算
bash batch_compute_faithfulness.sh
```

## 输出

- 单次: `faithfulness_exp/{method}/faithfulness_results_*.json`
- 批量: 编辑脚本中 `MODEL_NAMES`、`METHOD`、`MAX_ACTIONS`
