#!/bin/bash

# 运行5个模型的模拟
# 模型名称请在下面的MODELS数组中填写（留空供用户填写）

# 切换到脚本所在目录的上级目录的上级目录（TwinMarket根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT" || exit 1

# ==================== 配置参数 ====================
# 模型列表（用户需要在这里填写5个模型名称）
MODELS=(
    # "gpt-4o"  # 模型1
    # "llama-3.1-70b-instruct"  # 模型2
    # "anthropic/claude-sonnet-4.5"  # 模型3
    # "deepseek-ai/DeepSeek-V3.1"  # 模型4
    # "qwen3-235b-a22b-instruct-2507"  # 模型5
    "qwen-plus"
)

# 交易日范围（5个连续交易日）
START_DATE="2023-06-26"
END_DATE="2023-06-30"

# 数据库配置
USER_DB="data/sys_4.db"
NUM_USERS=4

# 激活概率（全部激活）
ACTIVATE_PROB=1.0

# 自动生成对手盘订单配置
# 当agent数量较少时（如4个），可以启用此选项以确保订单能够撮合
# 设置为 true 启用，false 禁用（默认）
AUTO_GENERATE_COUNTERPARTY="true"

# 输出目录
RESULTS_DIR="results"
ACCURACY_DIR="${RESULTS_DIR}/accuracy"

# ==================== 检查配置 ====================
# 检查是否有模型名称为空
EMPTY_MODELS=()
for i in "${!MODELS[@]}"; do
    if [ -z "${MODELS[$i]}" ]; then
        EMPTY_MODELS+=("模型$((i+1))")
    fi
done

if [ ${#EMPTY_MODELS[@]} -gt 0 ]; then
    echo "⚠️  警告: 以下模型名称未填写: ${EMPTY_MODELS[*]}"
    echo "   请在脚本中MODELS数组中填写模型名称"
    read -p "是否继续？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查数据库是否存在
if [ ! -f "$USER_DB" ]; then
    echo "❌ 用户数据库不存在: $USER_DB"
    echo "   请先运行: scripts/accuracy_exp/create_sys_4_db.sh"
    exit 1
fi

# 检查config/api.yaml是否存在
CONFIG_FILE="config/api.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# ==================== 备份原始配置 ====================
CONFIG_BACKUP="${CONFIG_FILE}.backup"
if [ ! -f "$CONFIG_BACKUP" ]; then
    cp "$CONFIG_FILE" "$CONFIG_BACKUP"
    echo "📋 已备份原始配置到: $CONFIG_BACKUP"
fi

# 创建输出目录
mkdir -p "$ACCURACY_DIR"

# ==================== 运行模拟 ====================
echo "=========================================="
echo "开始运行5个模型的模拟"
echo "=========================================="
echo "交易日范围: ${START_DATE} 到 ${END_DATE}"
echo "用户数量: ${NUM_USERS}"
echo "激活概率: ${ACTIVATE_PROB}"
echo "自动生成对手盘: ${AUTO_GENERATE_COUNTERPARTY}"
echo "结果目录: ${ACCURACY_DIR}"
echo "=========================================="
echo ""

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_INDEX=$((i+1))
    
    if [ -z "$MODEL" ]; then
        echo "⏭️  跳过模型${MODEL_INDEX}（未填写模型名称）"
        continue
    fi
    
    echo "=========================================="
    echo "运行模型 ${MODEL_INDEX}/5: ${MODEL}"
    echo "=========================================="
    
    # 设置环境变量：自动生成对手盘订单
    export AUTO_GENERATE_COUNTERPARTY="${AUTO_GENERATE_COUNTERPARTY}"
    if [ "${AUTO_GENERATE_COUNTERPARTY}" = "true" ]; then
        echo "✅ 已启用自动生成对手盘订单（适用于少量agent场景）"
    else
        echo "ℹ️  未启用自动生成对手盘订单（使用默认配置）"
    fi
    
    # 设置日志目录（去掉results/前缀，因为simulation.py会自动添加）
    LOG_DIR="${ACCURACY_DIR#results/}/${MODEL}"
    
    # 创建结果目录（simulation.py会自动创建，但我们需要先创建以复制数据库）
    FULL_RESULT_DIR="${RESULTS_DIR}/${LOG_DIR}"
    mkdir -p "$FULL_RESULT_DIR"
    
    # 复制数据库到结果目录
    USER_DB_PATH="${FULL_RESULT_DIR}/user_${NUM_USERS}.db"
    FORUM_DB_PATH="${FULL_RESULT_DIR}/forum_${NUM_USERS}.db"
    
    if [ ! -f "$USER_DB_PATH" ]; then
        cp "$USER_DB" "$USER_DB_PATH"
        echo "✅ 复制用户数据库到: $USER_DB_PATH"
    fi
    
    # 论坛数据库不需要复制，因为simulation.py会在非checkpoint模式下自动初始化
    # 如果论坛数据库已存在但可能损坏，删除它让simulation.py重新初始化
    if [ -f "$FORUM_DB_PATH" ]; then
        echo "⚠️  论坛数据库已存在，将在simulation.py中重新初始化"
        # 不在这里删除，让simulation.py的init_db_forum处理（它会删除并重新创建）
    fi
    
    # 更新配置文件中的模型名称
    python3 << EOF
import yaml
import sys

config_file = "${CONFIG_FILE}"
model_name = "${MODEL}"

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

config['model_name'] = model_name

with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"✅ 已更新配置文件中的模型名称: {model_name}")
EOF
    
    # 检查belief文件
    BELIEF_FILE="util/belief/belief_${NUM_USERS}.csv"
    if [ ! -f "$BELIEF_FILE" ]; then
        echo "⚠️  警告: Belief文件不存在: $BELIEF_FILE"
        echo "   尝试创建belief文件..."
        if [ -f "scripts/accuracy_exp/create_belief_${NUM_USERS}_csv.sh" ]; then
            bash "scripts/accuracy_exp/create_belief_${NUM_USERS}_csv.sh"
            if [ -f "$BELIEF_FILE" ]; then
                echo "✅ 已创建belief文件: $BELIEF_FILE"
                BELIEF_FILE_ARG="--belief_init_path ${BELIEF_FILE}"
            else
                echo "⚠️  创建失败，将使用默认belief配置（模拟可以继续，但可能缺少初始信念值）"
                BELIEF_FILE_ARG=""
            fi
        else
            echo "⚠️  找不到创建脚本，将使用默认belief配置（模拟可以继续，但可能缺少初始信念值）"
            BELIEF_FILE_ARG=""
        fi
    else
        BELIEF_FILE_ARG="--belief_init_path ${BELIEF_FILE}"
    fi
    
    # 生成时间戳
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${FULL_RESULT_DIR}/simulation_${TIMESTAMP}.log"
    
    # 运行模拟
    echo "🚀 开始运行模拟..."
    echo "   输出目录: ${FULL_RESULT_DIR}"
    echo "   日志文件: ${LOG_FILE}"
    echo ""
    
    python simulation.py \
        --log_dir "${LOG_DIR}" \
        --user_db "${USER_DB_PATH}" \
        --forum_db "${FORUM_DB_PATH}" \
        --user_graph_save_name "user_graph_${MODEL}" \
        --start_date "${START_DATE}" \
        --end_date "${END_DATE}" \
        --max_workers 256 \
        --config_path "${CONFIG_FILE}" \
        --activate_prob ${ACTIVATE_PROB} \
        ${BELIEF_FILE_ARG} \
        2>&1 | tee "${LOG_FILE}"
    
    SIMULATION_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $SIMULATION_EXIT_CODE -eq 0 ]; then
        echo "✅ 模型 ${MODEL} 的模拟完成！"
    else
        echo "❌ 模型 ${MODEL} 的模拟失败（退出码: $SIMULATION_EXIT_CODE）"
    fi
    
    echo ""
done

# ==================== 恢复原始配置 ====================
if [ -f "$CONFIG_BACKUP" ]; then
    cp "$CONFIG_BACKUP" "$CONFIG_FILE"
    echo "✅ 已恢复原始配置文件"
fi

echo "=========================================="
echo "所有模拟完成！"
echo "结果保存在: ${ACCURACY_DIR}/"
echo "=========================================="

