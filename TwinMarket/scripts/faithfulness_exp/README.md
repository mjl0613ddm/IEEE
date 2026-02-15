# Faithfulness实验脚本

本目录包含用于faithfulness实验的脚本。

## 脚本列表

1. **extract_action_features.py**: 提取action特征
2. **compute_faithfulness.py**: 计算faithfulness指标（deletion和insertion曲线）
3. **batch_compute_faithfulness.sh**: 批量计算faithfulness指标

## extract_action_features.py

从TwinMarket结果文件夹中提取每个(user_id, date)对的交易特征。

### 提取的特征

- `user_id`: 用户ID
- `date`: 日期
- `n_transactions`: 总交易笔数（buy + sell的sub_orders总数）
- `n_buy`: 买入笔数
- `n_sell`: 卖出笔数
- `buy_amount`: 买入总金额
- `sell_amount`: 卖出总金额
- `n_buy_stocks`: 买入涉及的股票数量
- `n_sell_stocks`: 卖出涉及的股票数量
- `target_date`: 风险发生的目标日期

### 使用方法

```bash
# 处理所有模型
python scripts/faithfulness_exp/extract_action_features.py

# 处理指定模型
python scripts/faithfulness_exp/extract_action_features.py --models gpt-4o-mini qwen-plus

# 处理指定结果目录
python scripts/faithfulness_exp/extract_action_features.py --results_dir /path/to/results

# 静默模式
python scripts/faithfulness_exp/extract_action_features.py --quiet
```

### 输出位置

对于每个 `{model}_{seed}` 文件夹，会在 `action_table/` 目录下生成：

- `action_table_{start_date}_{target_date}.csv`
- `action_table_{start_date}_{target_date}.json`

## compute_faithfulness.py

计算faithfulness指标，包括deletion和insertion曲线的AUC值。

### 功能

- **Deletion实验**: 从全部real action开始，逐步移除分数最高的action，计算风险变化
- **Insertion实验**: 从全部baseline action开始，逐步添加分数最高的action（并激活匹配订单），计算风险变化
- **AUC计算**: 计算归一化和原始AUC值
- **可视化**: 绘制deletion和insertion曲线图

### 使用方法

```bash
# 使用shapley方法，处理前10个action
python scripts/faithfulness_exp/compute_faithfulness.py \
  --log_dir results/gpt-4o-mini/gpt-4o-mini_42 \
  --method shapley \
  --max_actions 10

# 处理所有action
python scripts/faithfulness_exp/compute_faithfulness.py \
  --log_dir results/gpt-4o-mini/gpt-4o-mini_42 \
  --method shapley

# 详细输出模式
python scripts/faithfulness_exp/compute_faithfulness.py \
  --log_dir results/gpt-4o-mini/gpt-4o-mini_42 \
  --method shapley \
  --max_actions 10 \
  --verbose
```

### 输出位置

对于每个 `{model}_{seed}` 文件夹，会在 `faithfulness_exp/{method}/` 目录下生成：

- `faithfulness_results_{n}actions.json`: 结果JSON文件
- `faithfulness_curves_{n}actions.png`: 曲线图

### 输出格式

**JSON格式**：
```json
{
  "simulation_name": "gpt-4o-mini_42",
  "method": "shapley",
  "deletion_auc": 0.3008,
  "insertion_auc": 0.7632,
  "deletion_auc_raw": 0.0017,
  "insertion_auc_raw": 0.0045,
  "baseline_risk": 0.0,
  "real_risk": 0.1118,
  "deletion_curve": [0.1118, 0.0212, ...],
  "insertion_curve": [0.0, 0.0056, ...],
  "total_steps": 477,
  "truncated_steps": 10,
  "max_actions": 10
}
```

## batch_compute_faithfulness.sh

批量计算多个模型的faithfulness指标。

### 配置

编辑脚本中的配置参数：

```bash
# 模型名称列表
MODEL_NAMES=("gpt-4o-mini" "qwen-plus")

# 方法
METHOD="shapley"

# 最大action数量
MAX_ACTIONS=10
```

### 使用方法

```bash
# 运行批量计算
bash scripts/faithfulness_exp/batch_compute_faithfulness.sh
```

### 日志

日志文件保存在 `scripts/faithfulness_exp/logs/` 目录下，文件名包含时间戳。

## 注意事项

1. **Insertion实验的匹配订单功能**: 
   - 当添加一个action时，脚本可以自动查找并激活匹配的订单（同一天、同一股票、价格匹配的对手方订单）
   - 匹配逻辑：对于buy订单，查找价格匹配的sell订单；对于sell订单，查找价格匹配的buy订单
   - 价格匹配条件：buy价格 >= sell价格 * 0.95（允许5%的价格容差）
   - **重要问题**：启用匹配订单可能导致insertion曲线的风险值超过real_risk，这是因为：
     - 匹配的订单可能在其他日期也有action，导致激活了比全部real action更多的action
     - 累积效应：随着逐步添加action，匹配的订单越来越多，可能激活了远超预期的action数量
     - 某些action组合可能产生比全部real action更高的风险
   - **解决方案**：
     - **方案1（推荐）**：禁用匹配订单功能（在代码中设置`enable_matching_orders=False`），与EconAgent保持一致
     - **方案2**：优化匹配逻辑，限制每个action最多匹配的订单数量，或只匹配价格最接近的订单
   - **验证**：计算确实使用的是目标日期（target_date）的风险值，而不是整个时间序列的聚合值

2. **Action定义**: 一个action = (user_id, date)对，表示该用户在该日期的所有交易决策

3. **支持的方法**: 目前只实现了shapley方法，其他方法（random, llm, mast, loo）的接口已预留，待实现

4. **计算时间**: faithfulness实验需要运行多次反事实模拟，计算时间较长，建议先用较小的max_actions测试

5. **结果验证**:
   - Deletion曲线：应该从real_risk开始，逐步下降到接近baseline_risk
   - Insertion曲线：应该从baseline_risk开始，逐步上升到接近real_risk
   - 如果insertion曲线的值超过real_risk，可能的原因：
     - **匹配订单功能**：激活了过多不应该激活的订单（已修复：避免重复添加）
     - **风险指标计算问题**：少量交易时，VWAP可能不稳定，导致价格波动被放大
       - 已添加最小成交量阈值（MIN_VOLUME_THRESHOLD = 1000）
       - 当交易量小于阈值时，使用收盘价而不是VWAP计算市场平均价格
       - 这可以避免少量交易导致的风险指标异常
   - **建议**：如果发现insertion曲线异常，检查是否禁用了匹配订单功能，并确认风险指标计算是否正常
