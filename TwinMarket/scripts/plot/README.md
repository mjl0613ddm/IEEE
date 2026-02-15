# TwinMarket批量画图脚本

## 功能

为TwinMarket项目的每个模型和种子生成四张图表：
1. **风险曲线和累积Shapley值**：展示风险演化、累积Shapley值以及时间聚合的Shapley值
2. **Shapley值vs不稳定性散点图**：展示每个agent的Shapley绝对值与不稳定性的关系（C_ag指标）
3. **按agent聚合的Shapley值柱状图**：展示每个agent的风险贡献
4. **按behavior聚合的柱状图**：展示10种股票的风险贡献

## 使用方法

### 基本用法

```bash
cd /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket
python3 scripts/plot/batch_plot.py
```

### 参数说明

- `--base_path`：项目根路径（默认：`/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket`）
- `--models`：指定要处理的模型列表（默认：处理results/下所有模型）
- `--seeds`：指定要处理的种子列表（默认：处理所有种子，排除`*_rm`后缀）

### 示例

```bash
# 处理指定模型
python3 scripts/plot/batch_plot.py --models claude-3-haiku-20240307 gpt-4o-mini

# 处理指定种子
python3 scripts/plot/batch_plot.py --models claude-3-haiku-20240307 --seeds claude-3-haiku-20240307_42 claude-3-haiku-20240307_43
```

## 输出位置

生成的图表保存在：
```
results/{model}/{seed}/plot/
```

包含以下文件：
- `risk_and_cumulative_shapley.png`
- `shapley_instability_scatter.png`
- `agent_aggregated.png`
- `behavior_aggregated.png`
