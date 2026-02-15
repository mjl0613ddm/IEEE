#!/bin/bash
# 批量运行不同模型和种子的模拟，并生成图片
# 使用方法: ./batch_run_simulations.sh

# 注意：不使用 set -e，因为我们需要在单个运行失败时继续执行其他运行

# 设置工作目录
# 脚本在 EconAgent 根目录下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 配置（相对于项目根目录）
CONFIG_FILE="config.yaml"
SIMULATE_SCRIPT="simulate.py"
PLOT_SCRIPT="scripts/plot/plot_world_metrics.py"
DATA_ROOT="./data"

# 模型配置（模型名称 -> config.yaml中的模型标识符）
declare -A MODEL_MAP=(
    ["gpt"]="gpt-4o"
    ["llama"]="llama-3.1-70b-instruct"
    ["claude"]="anthropic/claude-sonnet-4.5"
)

# 种子列表（跳过42，因为已有结果）
SEEDS=(50 51 52 53 54)

# 备份原始配置文件
CONFIG_BACKUP="${CONFIG_FILE}.backup"
if [ ! -f "$CONFIG_BACKUP" ]; then
    cp "$CONFIG_FILE" "$CONFIG_BACKUP"
    echo "✅ 已备份配置文件: $CONFIG_BACKUP"
fi

# 函数：修改配置文件
update_config() {
    local model_name=$1
    local run_name=$2
    local seed=$3
    
    # 使用Python脚本来修改YAML文件（更安全）
    python3 << EOF
import yaml
import sys

config_file = "$CONFIG_FILE"
model_name = "$model_name"
run_name = "$run_name"
seed = int("$seed")

# 读取配置文件
with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 更新配置
config['llm']['model'] = model_name
config['simulation']['run_name'] = run_name
config['simulation']['seed'] = seed

# 写回文件
with open(config_file, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"✅ 已更新配置: model={model_name}, run_name={run_name}, seed={seed}")
EOF
}

# 函数：恢复配置文件
restore_config() {
    local quiet=${1:-0}  # 默认为0（不安静模式）
    if [ -f "$CONFIG_BACKUP" ]; then
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        if [ "$quiet" -eq 0 ]; then
            echo "✅ 已恢复配置文件"
        fi
    fi
}

# 函数：运行模拟
run_simulation() {
    local model_short=$1
    local seed=$2
    local run_name="${model_short}-${seed}"
    local success=0
    
    echo ""
    echo "=========================================="
    echo "🚀 开始运行: $run_name"
    echo "=========================================="
    
    # 更新配置
    local model_full="${MODEL_MAP[$model_short]}"
    update_config "$model_full" "$run_name" "$seed"
    
    # 运行模拟
    echo "📊 运行模拟..."
    if python3 "$SIMULATE_SCRIPT" --seed "$seed" --policy_model gpt; then
        echo "✅ 模拟完成: $run_name"
        success=1
    else
        echo "❌ 模拟失败: $run_name"
        success=0
    fi
    
    # 生成图片（即使模拟失败也尝试生成，以防部分数据已生成）
    if [ $success -eq 1 ]; then
        echo "📈 生成图片..."
        local data_folder="${DATA_ROOT}/${run_name}"
        
        if [ -d "$data_folder" ]; then
            # 检查CSV文件是否存在
            local csv_file="${data_folder}/metrics_csv/world_metrics.csv"
            if [ -f "$csv_file" ]; then
                # 生成 price_inflation_rate 图片
                python3 "$PLOT_SCRIPT" --csv-file "$csv_file" \
                    price_inflation_rate 2>/dev/null || echo "⚠️  生成 price_inflation_rate 图片失败"
                
                # 生成 risk_indicator_naive 图片
                python3 "$PLOT_SCRIPT" --csv-file "$csv_file" \
                    risk_indicator_naive 2>/dev/null || echo "⚠️  生成 risk_indicator_naive 图片失败"
                
                echo "✅ 图片生成完成: $run_name"
            else
                echo "⚠️  警告: CSV文件不存在: $csv_file"
            fi
        else
            echo "⚠️  警告: 数据目录不存在: $data_folder"
        fi
    fi
    
    # 恢复配置文件（每次运行后都恢复，避免配置被破坏）
    restore_config 1  # 安静模式
    
    if [ $success -eq 1 ]; then
        echo "✅ 完成: $run_name"
    else
        echo "❌ 失败: $run_name"
    fi
    echo ""
    
    return $success
}

# 主循环
echo "=========================================="
echo "批量运行模拟脚本"
echo "=========================================="
echo "模型: gpt, llama, claude"
echo "种子: ${SEEDS[@]}"
echo ""

# 捕获中断信号，确保恢复配置
trap restore_config EXIT INT TERM

# 运行所有组合
failed_runs=()
for model_short in gpt llama claude; do
    for seed in "${SEEDS[@]}"; do
        if ! run_simulation "$model_short" "$seed"; then
            failed_runs+=("${model_short}-${seed}")
        fi
    done
done

# 最终恢复配置文件（确保恢复）
restore_config

# 报告结果
echo ""
if [ ${#failed_runs[@]} -eq 0 ]; then
    echo "✅ 所有运行都成功完成！"
else
    echo "⚠️  以下运行失败:"
    for run in "${failed_runs[@]}"; do
        echo "   - $run"
    done
fi

echo "=========================================="
echo "🎉 所有任务完成！"
echo "=========================================="

