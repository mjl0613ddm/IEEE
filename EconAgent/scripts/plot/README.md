# 批量绘图脚本使用说明

## 功能

为ACL24-EconAgent项目的每个模型和种子批量生成四张图表：

1. **风险曲线和累积shapley value** - 双子图
2. **Shapley绝对值与Instability散点图** - 展示C_ag指标
3. **按Agent聚合的Shapley值** - 柱状图
4. **按Behavior聚合的Shapley值** - 柱状图

## 使用方法

### 基本用法（处理所有模型和种子）

```bash
cd /mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent
python3 scripts/plot/batch_plot.py
```

### 指定模型

```bash
python3 scripts/plot/batch_plot.py --models claude gpt
```

### 指定种子

```bash
python3 scripts/plot/batch_plot.py --models claude --seeds claude_42 claude_44
```

## 参数说明

- `--base_path`: 项目根目录路径（默认：`/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent`）
- `--models`: 要处理的模型列表（默认：处理datas/目录下的所有模型）
- `--seeds`: 要处理的种子列表（默认：处理所有非`*_rm`后缀的种子）

## 输出位置

生成的图片保存在：`datas/{model}/{seed}/plot/`

**注意**：脚本会自动排除以`_rm`后缀结尾的种子目录。
