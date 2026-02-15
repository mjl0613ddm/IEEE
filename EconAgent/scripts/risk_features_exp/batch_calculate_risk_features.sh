#!/bin/bash
# EconAgent 风险特征批量计算Shell脚本包装器

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/batch_calculate_risk_features.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# 运行Python脚本，传递所有参数
python3 "$PYTHON_SCRIPT" "$@"
