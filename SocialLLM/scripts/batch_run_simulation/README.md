# Batch Simulation (SocialLLM)

批量运行 SocialLLM 模拟。

## 用法

1. 编辑 `batch_run_simulation.sh`：
   ```bash
   MODEL_NAMES=("gpt-4o-mini" "qwen-plus")
   NUM_AGENTS=20
   NUM_STEPS=30
   MAX_CONCURRENT=10
   ```

2. 准备 `config/api_{model}.yaml`（参考 `config/api_example.yaml`）

3. 运行：
   ```bash
   bash scripts/batch_run_simulation/batch_run_simulation.sh
   ```

## 输出

- `results/{model}/{model}_{seed}/actions.json`, `results.json`, `random_states.json`
- 日志: `scripts/batch_run_simulation/logs/`
