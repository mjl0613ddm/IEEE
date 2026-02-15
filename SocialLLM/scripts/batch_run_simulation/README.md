# 批量运行SocialLLM模拟脚本

## 简介

`batch_run_simulation.sh` 是一个用于批量运行SocialLLM模拟的bash脚本。它支持：

- 多模型批量运行
- 每个模型使用不同的seed运行多次
- 并发控制（适合API调用）
- 自动跳过已有结果
- 完整的日志记录

## 功能特性

1. **多模型支持**：可以同时运行多个模型的模拟
2. **多seed运行**：每个模型使用不同的seed运行指定次数（默认15次，seed 42-56）
3. **并发控制**：支持可配置的并发数（默认10），适合I/O密集型的API调用
4. **智能跳过**：自动检测并跳过已有结果的模拟，避免重复运行
5. **完整日志**：所有操作都有详细的时间戳日志记录

## 使用方法

### 1. 配置模型列表

编辑脚本文件，修改 `MODEL_NAMES` 变量：

```bash
MODEL_NAMES=("gpt-4o-mini" "qwen-plus" "claude-3-haiku-20240307")
```

### 2. 准备API配置文件

为每个模型创建对应的API配置文件，放在 `config/` 目录下：

- `config/api_gpt-4o-mini.yaml`
- `config/api_qwen-plus.yaml`
- `config/api_claude-3-haiku-20240307.yaml`
- ...

配置文件格式参考 `config/api_example.yaml`：

```yaml
api_key:
  - your-api-key-here
model_name: gpt-4o-mini
base_url: http://your-api-server/v1
```

### 3. 配置参数（可选）

在脚本中修改以下参数：

```bash
# 固定参数
NUM_AGENTS=20          # Agent数量
NUM_STEPS=30           # 模拟步数
SEED_START=42          # 起始seed
SEED_COUNT=15          # 运行次数（seed范围：SEED_START 到 SEED_START+SEED_COUNT-1）

# 并发数（API调用是I/O密集型，与CPU核心数无关）
MAX_CONCURRENT=10      # 默认并发数，可根据API服务器容量调整
```

### 4. 运行脚本

```bash
cd SocialLLM  # or path to SocialLLM directory
bash scripts/batch_run_simulation/batch_run_simulation.sh
```

## 输出结构

### 结果目录

所有结果保存在 `results/{model_name}/` 目录下：

```
results/
├── gpt-4o-mini/
│   ├── gpt-4o-mini_42/
│   │   ├── actions.json          # 动作历史
│   │   ├── random_states.json    # 随机状态
│   │   └── results.json          # 模拟结果（包含每个时间步的belief和风险）
│   ├── gpt-4o-mini_43/
│   │   └── ...
│   └── ...
├── qwen-plus/
│   ├── qwen-plus_42/
│   │   └── ...
│   └── ...
└── ...
```

### 日志文件

日志文件保存在 `scripts/batch_run_simulation/logs/` 目录下：

```
logs/
└── batch_run_simulation_20260110_105323.log
```

日志包含：
- 每个任务的启动和完成时间
- 成功/失败/跳过状态
- 错误信息（如果有）
- 总体统计信息

## 跳过已有结果

脚本会自动检查结果目录中是否存在 `results.json` 文件：

- **如果存在**：跳过该任务，不重新运行，计入成功数
- **如果不存在**：正常运行模拟

这样可以：
- 避免重复运行已完成的模拟
- 支持断点续跑（脚本中断后重新运行会继续未完成的任务）
- 节省时间和API调用成本

## 并发控制

脚本使用分批处理的方式控制并发：

1. 每次启动 `MAX_CONCURRENT` 个任务（默认10个）
2. 等待这批任务全部完成后，再启动下一批
3. 适合I/O密集型的API调用，不会占用过多CPU资源

**调整并发数**：
- 如果API服务器容量大，可以增加 `MAX_CONCURRENT`
- 如果API服务器有速率限制，可以减少 `MAX_CONCURRENT`
- 建议值：5-20（根据API服务器实际情况调整）

## 示例

### 运行单个模型

修改脚本中的 `MODEL_NAMES`：

```bash
MODEL_NAMES=("gpt-4o-mini")
```

### 运行多个模型

```bash
MODEL_NAMES=("gpt-4o-mini" "qwen-plus" "claude-3-haiku-20240307")
```

### 修改运行次数

```bash
SEED_START=42
SEED_COUNT=20  # 运行20次（seed 42-61）
```

### 调整并发数

```bash
MAX_CONCURRENT=5  # 降低并发数，适合API速率限制较严格的情况
```

## 注意事项

1. **API配置文件**：确保每个模型都有对应的API配置文件，且路径正确
2. **网络连接**：确保能够访问API服务器
3. **磁盘空间**：确保有足够的磁盘空间保存结果
4. **Python环境**：确保Python环境已安装所需依赖（yaml等）
5. **权限**：确保脚本有执行权限（`chmod +x batch_run_simulation.sh`）

## 故障排查

### 问题：API配置文件不存在

**错误信息**：
```
ERROR: API配置文件不存在: config/api_xxx.yaml，跳过
```

**解决方法**：
- 检查模型名称是否正确
- 确认API配置文件是否存在且命名正确（`api_{model_name}.yaml`）

### 问题：Python模块缺失

**错误信息**：
```
ModuleNotFoundError: No module named 'yaml'
```

**解决方法**：
```bash
pip install pyyaml
```

### 问题：结果目录权限错误

**错误信息**：
```
无法创建结果目录
```

**解决方法**：
- 检查目录权限
- 确保有写入权限

## 相关文件

- `main.py`: 主程序脚本
- `config/config.yaml`: 主配置文件
- `config/api_example.yaml`: API配置文件示例
- `scripts/batch_run_simulation/logs/`: 日志目录

## 更新日志

- 2026-01-10: 初始版本
  - 支持多模型批量运行
  - 支持并发控制
  - 支持自动跳过已有结果
