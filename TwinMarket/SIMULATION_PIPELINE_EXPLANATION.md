# TwinMarket Simulation Pipeline 详细说明

## 📋 整体流程概览

这是一个**股票市场模拟系统**，用于计算每个交易者（agent）对市场风险的贡献度（Shapley Value）。整个系统模拟了真实的股票交易流程。

## 🔄 完整流程步骤

### 1. **数据准备阶段**

#### 1.1 数据库的作用
数据库（SQLite）存储了三个核心表：
- **StockData**: 股票历史数据（价格、成交量、技术指标等）
- **Profiles**: 用户档案（持仓、收益、历史交易等）
- **TradingDetails**: 交易明细记录

**为什么需要数据库？**
- 存储历史状态：每个交易日需要知道前一天的收盘价、用户持仓等
- 状态更新：撮合交易后会更新股票价格、用户持仓等
- 数据持久化：支持多日连续模拟，状态在日期间传递

#### 1.2 决策文件（JSON）
每个交易日有一个JSON文件，格式如下：
```json
{
  "user_id_1": {
    "stock_decisions": {
      "STOCK_CODE": {
        "action": "buy",  // 或 "sell", "hold"
        "cur_position": 1000,  // 当前持仓
        "target_position": 2000,  // 目标持仓
        "sub_orders": [
          {"price": 13.24, "quantity": 84800}
        ]
      }
    }
  }
}
```

### 2. **单日交易流程（test_matching_system）**

#### 2.1 读取初始状态
```
1. 连接数据库，读取：
   - StockData: 获取所有股票的历史价格数据
   - StockProfile: 股票基本信息
   - Profiles: 用户当前持仓和收益

2. 读取决策JSON文件：
   - 加载当天所有用户的交易决策
```

#### 2.2 生成股票初始价格
```python
generate_stock_data(decisions, df_stock, current_date)
```
- 从数据库中获取每只股票**前一天的收盘价**
- 作为当天撮合的起始价格

#### 2.3 转换决策为订单
```python
create_orders_from_decisions(decisions, current_date)
```
- 将JSON中的决策转换为标准订单对象（Order）
- 为每个订单分配随机时间戳（模拟真实交易时间）
- 时间戳分布在：上午9:30-11:30，下午13:00-15:00

#### 2.4 订单撮合（核心）
```python
process_daily_orders(orders, last_prices, current_date, ...)
```

**撮合逻辑：**
1. **按股票分组**：将所有订单按股票代码分组
2. **订单复制**：如果买卖不平衡（比例>2.5），会复制订单以平衡市场
3. **时间排序**：按时间戳排序所有订单
4. **撮合执行**：
   - 遍历每个订单
   - 找到匹配的对手单（价格匹配）
   - 执行交易，更新价格
   - 价格会根据买卖压力动态调整

**价格形成机制：**
- 初始价格 = 前一天收盘价
- 每笔交易后，价格会根据买卖订单的量和价进行调整
- 最终收盘价 = 最后一笔交易的价格

#### 2.5 更新数据库
交易完成后，更新三个表：

**a) StockData表更新**
```python
update_stock_data_table(...)
```
- 更新收盘价、涨跌幅
- 计算技术指标（移动平均、MACD等）
- 更新成交量数据

**b) TradingDetails表更新**
```python
update_trading_details_table(...)
```
- 记录每笔交易的详细信息
- 包括：交易时间、价格、数量、买卖双方等

**c) Profiles表更新**
```python
update_profiles_table(...)
```
- 更新每个用户的持仓数量
- 更新用户收益（基于新的股票价格）
- 更新用户总资产

### 3. **反事实模拟（Counterfactual Simulation）**

这是Shapley Value计算的核心！

#### 3.1 问题：为什么要复制数据库？

**原因：每个反事实模拟需要独立的状态**

假设我们要计算：如果用户A使用baseline策略（固定动作），市场风险会是多少？

流程：
1. **复制数据库** → 创建独立的状态副本
2. **修改决策** → 将用户A的决策改为baseline
3. **运行撮合** → 用修改后的决策运行完整交易流程
4. **计算风险** → 基于新的价格序列计算风险指标

**为什么不能共享数据库？**
- `test_matching_system`会**修改数据库**（更新价格、持仓等）
- 如果多个反事实模拟共享同一个数据库，它们会互相干扰
- 每个模拟需要从**相同的初始状态**开始，但产生**不同的结果**

#### 3.2 Baseline策略（固定动作）

有两种baseline类型：

**a) `no_action`（无动作）**
```python
# 用户不进行任何交易
counterfactual[user_id] = {
    "stock_decisions": {}  # 空决策
}
```

**b) `hold`（持有）**
```python
# 用户保持当前持仓，不买卖
counterfactual[user_id] = {
    "stock_decisions": {
        "STOCK_CODE": {
            "action": "hold",  # 标记为hold
            "cur_position": 1000,  # 保持当前持仓
            "sub_orders": []  # 无订单
        }
    }
}
```

#### 3.3 反事实模拟流程

```python
def run_counterfactual_simulation(...):
    # 1. 复制数据库（确保独立状态）
    copy_and_prepare_database(user_db, temp_db_path, ...)
    
    # 2. 遍历每个交易日
    for date in all_dates:
        # 2.1 加载真实决策
        real_decisions = load_real_decisions(log_dir, date)
        
        # 2.2 创建反事实决策
        # - active_users: 使用真实决策的用户
        # - 其他用户: 使用baseline策略
        counterfactual_decisions = create_counterfactual_decisions(
            date, active_users_for_date, real_decisions, baseline_type
        )
        
        # 2.3 保存到临时JSON文件
        save_counterfactual_decisions(counterfactual_decisions, temp_json_file)
        
        # 2.4 运行撮合引擎（使用复制的数据库）
        test_matching_system(
            current_date=date,
            json_file_path=temp_json_file,
            db_path=temp_db_path,  # 使用复制的数据库
            base_path=temp_base_path
        )
        # 注意：这会修改temp_db_path中的数据库！
    
    # 3. 计算风险指标
    # 从更新后的数据库中读取价格序列，计算风险
    risk_value = calculate_risk_metric(...)
```

### 4. **Shapley Value计算**

#### 4.1 蒙特卡洛采样
```python
for sample_idx in range(n_samples):
    # 随机排列所有players
    permuted_players = shuffle(all_players)
    
    # 逐个添加player，计算边际贡献
    current_subset = set()
    prev_risk = baseline_risk
    
    for player in permuted_players:
        current_subset.add(player)
        # 计算当前子集的风险
        current_risk = calculate_risk_for_subset(
            current_subset, ...  # 这个函数会运行反事实模拟！
        )
        # 边际贡献 = 当前风险 - 之前风险
        marginal_contrib = current_risk - prev_risk
        shapley_values[player] += marginal_contrib
        prev_risk = current_risk
```

#### 4.2 为什么需要这么多数据库复制？

**每个`calculate_risk_for_subset`调用都会：**
1. 复制数据库（创建独立状态）
2. 运行多日反事实模拟
3. 修改数据库（更新价格、持仓等）
4. 计算风险指标

**如果有100个players，1000次采样：**
- 理论最大调用次数：100 × 1000 = 100,000次
- 实际由于缓存会少一些，但仍然很多

## 🎯 关键理解点

### 1. **数据库是状态存储**
- 不是只读的！每次撮合都会修改数据库
- 每个反事实模拟需要从**相同的初始状态**开始
- 所以必须复制数据库，确保状态独立

### 2. **固定动作（Baseline）的含义**
- `no_action`: 用户完全不参与交易
- `hold`: 用户保持当前持仓，不买卖
- 目的是计算：如果用户不按真实决策行动，市场会怎样？

### 3. **为什么不能固定一个数据库？**
- 因为`test_matching_system`会**修改**数据库
- 如果多个模拟共享数据库，它们会互相干扰
- 每个模拟需要独立的状态演化

### 4. **优化方向**
- **轻量级模式**（`use_lightweight=True`）：
  - 不运行完整撮合系统
  - 直接用公式计算价格，不需要数据库
  - 速度快，但只支持部分指标

- **数据库快照/事务回滚**：
  - 使用数据库快照机制，减少复制开销
  - 或使用事务回滚，每次模拟后回滚到初始状态

## 📊 完整流程图

```
原始数据库 (user_db)
    ↓
[Shapley计算开始]
    ↓
[对于每个采样]
    ↓
[对于每个player子集]
    ↓
    复制数据库 → temp_db_path
    ↓
    [对于每个交易日]
        ↓
        加载真实决策
        ↓
        创建反事实决策（部分用户用baseline）
        ↓
        运行撮合引擎 → 修改temp_db_path
        ↓
        更新股票价格、用户持仓等
    ↓
    计算风险指标（基于更新后的数据库）
    ↓
    计算边际贡献
    ↓
[累加Shapley值]
```

## 💡 总结

**数据库的作用：**
- 存储市场状态（价格、持仓、历史数据）
- 支持状态演化（每次撮合后更新状态）
- 支持多日连续模拟（状态在日期间传递）

**为什么复制数据库：**
- 每个反事实模拟需要独立的状态
- 撮合引擎会修改数据库
- 必须从相同初始状态开始，但产生不同结果

**固定动作（Baseline）的流程：**
- 将部分用户的真实决策替换为baseline策略
- 运行完整撮合流程
- 计算新的市场风险
- 用于计算该用户对风险的贡献

---

## 🚀 不经过数据库的方法

### 方法1：轻量级模式（已实现，推荐）

**使用方式：**
```bash
python calculate_shapley_attribution.py --lightweight ...
```

**工作原理：**
1. **完全跳过数据库和撮合系统**
2. **直接从JSON决策文件计算价格**：
   - 读取每个交易日的决策JSON文件
   - 提取买卖订单信息
   - 使用价格冲击模型直接计算价格变化：
     ```
     价格变化 = (买入量 - 卖出量) / 总成交量 × 冲击因子 × sqrt(归一化成交量)
     新价格 = 旧价格 × (1 + 价格变化)
     ```
3. **计算风险指标**：基于价格序列直接计算

**优点：**
- ✅ **完全不需要数据库**：只读取JSON文件
- ✅ **速度快**：比完整撮合快10-100倍
- ✅ **无I/O瓶颈**：不需要复制数据库
- ✅ **内存占用小**：只存储价格字典

**限制：**
- ⚠️ **只支持 `risk_indicator_simple` 指标**
- ⚠️ **价格计算是近似值**：不运行完整撮合，价格可能不够精确
- ⚠️ **不支持需要交易明细的指标**：如成交量、VWAP等

**适用场景：**
- 只需要计算 `risk_indicator_simple` 指标
- 对价格精度要求不高
- 需要快速计算大量Shapley值

### 方法2：扩展轻量级模式（需要开发）

**思路：**
在轻量级模式基础上，生成简化的 `daily_summary` 和 `transactions` CSV文件，支持更多风险指标。

**实现要点：**
1. **生成daily_summary.csv**：
   ```python
   # 从计算出的价格生成
   daily_summary = {
       'date': date,
       'stock_code': stock_code,
       'closing_price': calculated_price,
       'volume': estimated_volume,  # 从订单量估算
       'pre_close': previous_price,
       'change': price_change,
       'pct_chg': price_change_pct
   }
   ```

2. **生成transactions.csv**：
   ```python
   # 从订单信息生成（简化版）
   transactions = {
       'timestamp': order_timestamp,
       'stock_code': stock_code,
       'executed_price': estimated_price,  # 使用计算出的价格
       'executed_quantity': order_quantity,
       'direction': order_direction
   }
   ```

3. **然后使用RiskMetricsCalculator计算其他指标**

**优点：**
- ✅ 支持更多风险指标
- ✅ 仍然不需要数据库
- ✅ 比完整撮合快

**缺点：**
- ⚠️ 需要开发工作
- ⚠️ 交易明细是估算的，可能不够准确

### 方法3：内存数据库（需要改造）

**思路：**
使用SQLite内存数据库（`:memory:`），避免文件I/O。

**实现要点：**
```python
# 创建内存数据库
memory_db = sqlite3.connect(':memory:')

# 从原始数据库复制结构到内存
source_conn = sqlite3.connect(user_db)
source_conn.backup(memory_db)

# 使用内存数据库运行撮合
test_matching_system(..., db_path=':memory:')
```

**优点：**
- ✅ 不需要复制文件
- ✅ 支持完整撮合系统
- ✅ 支持所有风险指标

**缺点：**
- ⚠️ 需要改造代码（当前代码使用文件路径）
- ⚠️ 内存占用较大
- ⚠️ 多线程时每个线程需要独立内存数据库

### 方法4：数据库快照/事务回滚（需要改造）

**思路：**
使用数据库快照或事务回滚，每次模拟后回滚到初始状态。

**实现要点：**
```python
# 使用事务
conn = sqlite3.connect(db_path)
conn.execute("BEGIN TRANSACTION")
# ... 运行撮合 ...
conn.execute("ROLLBACK")  # 回滚到初始状态
```

**优点：**
- ✅ 只需要一个数据库副本
- ✅ 支持完整撮合系统

**缺点：**
- ⚠️ SQLite的ROLLBACK可能不够彻底（WAL模式问题）
- ⚠️ 需要确保所有操作都在事务内
- ⚠️ 多线程时可能有锁竞争

## 📊 方法对比

| 方法 | 需要数据库 | 支持指标 | 速度 | 实现难度 | 推荐度 |
|------|-----------|---------|------|---------|--------|
| **轻量级模式** | ❌ 不需要 | `risk_indicator_simple` | ⚡⚡⚡ 很快 | ✅ 已实现 | ⭐⭐⭐⭐⭐ |
| **扩展轻量级** | ❌ 不需要 | 多个指标 | ⚡⚡ 快 | 🛠️ 需开发 | ⭐⭐⭐⭐ |
| **内存数据库** | ✅ 需要 | 所有指标 | ⚡⚡ 快 | 🛠️ 需改造 | ⭐⭐⭐ |
| **事务回滚** | ✅ 需要 | 所有指标 | ⚡ 中等 | 🛠️ 需改造 | ⭐⭐ |
| **完整撮合（当前）** | ✅ 需要 | 所有指标 | 🐌 慢 | ✅ 已实现 | ⭐ |

## 🎯 推荐方案

**如果只需要 `risk_indicator_simple` 指标：**
- ✅ **直接使用轻量级模式**（`--lightweight`）
- 完全不需要数据库，速度最快

**如果需要其他风险指标：**
1. **短期**：继续使用完整撮合系统（虽然慢，但准确）
2. **长期**：开发扩展轻量级模式，生成简化的CSV文件支持更多指标

**如果必须使用完整撮合但想优化：**
- 考虑使用内存数据库或数据库快照机制

