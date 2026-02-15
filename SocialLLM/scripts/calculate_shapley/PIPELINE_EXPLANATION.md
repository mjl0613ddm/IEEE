# SocialLLM Shapley Value 计算 Pipeline 和反事实模拟逻辑

## 整体流程

### 1. 计算目标
- 对于每个 `(agent_id, timestep)` 组合，计算Shapley value
- 所有Shapley值的总和应该等于 `max_risk - initial_risk`
- 用于归因分析：每个agent在每个时间步对最高风险点的贡献

### 2. Shapley Value定义
使用蒙特卡洛方法计算Shapley value：

```
phi_i(t) = (1/n_samples) * sum over all samples [marginal_contribution(S, i, t)]
```

其中 `marginal_contribution(S, i, t) = v(S) - v(S ∪ {i})`

## 边际贡献计算（calculate_marginal_contribution）

### 输入参数
- `agent_id`: Agent ID (i)
- `timestep`: 时间步 (t)
- `subset_S`: 子集S（不包含agent_id）
- `target_timestep`: 目标时间步（max_risk_timestep）

### 计算过程

1. **计算 v(S)**：
   - Mask掉所有不在S中的agent在时间步t的动作
   - 运行反事实模拟到 `target_timestep`
   - 返回 `target_timestep` 的风险值
   - 含义：只有S中的agent在时间步t行动时，在target_timestep的风险值

2. **计算 v(S ∪ {i})**：
   - Mask掉所有不在S ∪ {i}中的agent在时间步t的动作
   - 运行反事实模拟到 `target_timestep`
   - 返回 `target_timestep` 的风险值
   - 含义：只有S ∪ {i}中的agent在时间步t行动时，在target_timestep的风险值

3. **计算边际贡献**：
   ```python
   marginal = v(S) - v(S ∪ {i})
   ```
   - 如果agent i的行动增加风险：v(S ∪ {i}) > v(S)，则marginal < 0
   - 如果agent i的行动减少风险：v(S ∪ {i}) < v(S)，则marginal > 0

## 反事实模拟（run_counterfactual_with_masked_set_optimized）

### 输入参数
- `initial_beliefs`: 初始belief值列表（从results.json的timestep 0读取）
- `masked_actions`: 被mask的动作集合 `{(agent_id, timestep)}`
- `target_timestep`: 目标时间步（max_risk_timestep）

### 模拟过程

1. **初始化**：
   - 创建CounterfactualModel（rule-based，不使用LLM）
   - 设置所有agent的初始belief值为 `initial_beliefs`
   - 设置masked_actions

2. **运行模拟**：
   - 从timestep 0开始，运行到 `target_timestep`（优化：不运行完整模拟）
   - 对于每个时间步：
     - 如果 `(agent_id, timestep)` 在 `masked_actions` 中：
       - 该agent在该时间步不发帖
       - 该agent在该时间步不查看任何帖子
       - 该agent在该时间步不互动（不点赞/点踩）
       - 其他agent在该时间步看不到该agent之前发的帖子
     - 否则：正常行动（rule-based决策）

3. **返回结果**：
   - 返回 `target_timestep` 的风险值

### 重要特性

- **完全独立**：每个反事实模拟都从 `initial_beliefs` 开始，不依赖原始的actions.json和random_states.json
- **Rule-based**：使用agent的belief值和偏好进行决策（基于概率分布）
- **随机性**：每次模拟的随机数可能不同（导致结果略有差异）
- **优化**：只运行到 `target_timestep`，不运行完整模拟

## Mask逻辑（counterfactual.py中的step函数）

### Masked Agent的行为

如果 `(agent_id, timestep)` 在 `masked_actions` 中：

1. **发帖阶段**：
   - 强制不发帖（`should_post = False`）

2. **查看帖子阶段**：
   - 不查看任何帖子
   - 不进行任何互动（不点赞/点踩）

3. **帖子可见性**：
   - 其他agent在该时间步看不到该agent之前发的帖子
   - 实现方式：在构建可用帖子列表时，过滤掉被mask的agent的帖子

### 实现细节

```python
# 在step函数中
for agent in self.agents:
    action_key = (agent.agent_id, self.timestep)
    
    if action_key in self.masked_actions:
        should_post = False  # 不发帖
    else:
        should_post = self.rule_based_decide_post(agent)  # 正常决策
    
    # 查看帖子时，过滤掉被mask的agent的帖子
    available_post_ids = [post_id for post_id in ... 
                          if post["author_id"] not in masked_agent_ids_at_current_timestep]
```

## 计算流程示例

假设我们要计算agent 0在时间步5的Shapley value：

1. **蒙特卡洛采样**（n_samples=1000次）：
   - 对于每次采样：
     - 随机选择子集S（例如：S = {1, 3, 5}，不包含agent 0）
     - 计算v(S)：
       - Mask掉agent 0, 2, 4在时间步5的动作
       - 只有agent 1, 3, 5在时间步5行动
       - 运行反事实模拟到target_timestep，得到风险值
     - 计算v(S ∪ {0})：
       - Mask掉agent 2, 4在时间步5的动作
       - 只有agent 0, 1, 3, 5在时间步5行动
       - 运行反事实模拟到target_timestep，得到风险值
     - 边际贡献 = v(S) - v(S ∪ {0})

2. **计算Shapley value**：
   - Shapley value = 所有边际贡献的平均值

3. **验证**：
   - 所有Shapley值的总和应该接近 `max_risk - initial_risk`

## 关键点总结

1. **边际贡献的定义**：
   - `marginal = v(S) - v(S ∪ {i})`
   - 这是修复后的版本（已取反）

2. **v(S)的含义**：
   - v(S) = 当只有S中的agent在时间步t行动时，在target_timestep的风险值
   - 注意：所有agent都参与模拟，但在时间步t只有S中的agent有行动

3. **反事实模拟的特性**：
   - 完全独立（从initial_beliefs开始）
   - Rule-based决策
   - 只运行到target_timestep（优化）

4. **Mask逻辑**：
   - Mask掉(agent_id, timestep)意味着该agent在该时间步不参与（不发帖、不查看、不互动）
