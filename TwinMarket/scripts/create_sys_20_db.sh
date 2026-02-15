#!/bin/bash

# 从 sys_1000.db 中提取20个用户创建 sys_20.db
# 支持随机选择或顺序选择

SOURCE_DB="data/sys_1000.db"
TARGET_DB="data/sys_20.db"
NUM_USERS=20

# 默认使用随机选择，可以通过参数 --sequential 改为顺序选择
RANDOM_SELECT=true
if [ "$1" == "--sequential" ]; then
    RANDOM_SELECT=false
    echo "=== 使用顺序选择模式 ==="
else
    echo "=== 使用随机选择模式（推荐） ==="
    echo "   如需顺序选择，请使用: $0 --sequential"
fi

echo "=== 从 ${SOURCE_DB} 创建 ${TARGET_DB} ==="

# 检查源数据库是否存在
if [ ! -f "$SOURCE_DB" ]; then
    echo "❌ 源数据库不存在: $SOURCE_DB"
    exit 1
fi

# 删除目标数据库（如果存在）
if [ -f "$TARGET_DB" ]; then
    echo "⚠️  目标数据库已存在，将删除: $TARGET_DB"
    rm "$TARGET_DB"
fi

# 复制数据库结构
echo "📋 复制数据库结构..."
sqlite3 "$SOURCE_DB" ".schema" | sqlite3 "$TARGET_DB"

# 创建临时SQL文件
TEMP_SQL=$(mktemp)

# 根据选择模式决定排序方式
if [ "$RANDOM_SELECT" = true ]; then
    ORDER_CLAUSE="ORDER BY RANDOM()"
    echo "🎲 随机选择 ${NUM_USERS} 个用户..."
else
    ORDER_CLAUSE=""
    echo "📋 顺序选择前 ${NUM_USERS} 个用户..."
fi

cat > "$TEMP_SQL" <<EOF
-- 复制用户数据
ATTACH DATABASE '${SOURCE_DB}' AS source_db;

-- 首先创建一个临时表存储选择的用户ID，确保所有表使用相同的用户集合
CREATE TEMP TABLE selected_users AS
SELECT user_id FROM source_db.Profiles 
${ORDER_CLAUSE}
LIMIT ${NUM_USERS};

-- 复制 Profiles 表（使用选择的用户ID）
INSERT INTO Profiles 
SELECT p.* FROM source_db.Profiles p
INNER JOIN selected_users s ON p.user_id = s.user_id;

-- 复制 Strategy 表（使用相同的用户ID，确保一致性）
INSERT INTO Strategy 
SELECT s.* FROM source_db.Strategy s
INNER JOIN selected_users u ON s.user_id = u.user_id;

-- 复制 StockData 表（所有股票数据）
INSERT INTO StockData 
SELECT * FROM source_db.StockData;

-- 复制 StockProfile 表（所有股票资料）
INSERT INTO StockProfile 
SELECT * FROM source_db.StockProfile;

-- 复制 TradingDetails 表（对应选择的20个用户的交易记录）
INSERT INTO TradingDetails 
SELECT t.* FROM source_db.TradingDetails t
INNER JOIN selected_users u ON t.user_id = u.user_id;

-- 清理临时表
DROP TABLE selected_users;

DETACH DATABASE source_db;
EOF

# 执行SQL
echo "💾 复制数据..."
sqlite3 "$TARGET_DB" < "$TEMP_SQL"

# 清理临时文件
rm "$TEMP_SQL"

# 验证结果
USER_COUNT=$(sqlite3 "$TARGET_DB" "SELECT COUNT(DISTINCT user_id) FROM Profiles;")

# 统计分布情况
STRATEGY_DIST=$(sqlite3 "$TARGET_DB" "SELECT strategy, COUNT(*) FROM (SELECT p.user_id, s.strategy FROM Profiles p JOIN Strategy s ON p.user_id = s.user_id) GROUP BY strategy;" | tr '\n' ' ')
CASH_DIST=$(sqlite3 "$TARGET_DB" "SELECT ini_cash/10000, COUNT(*) FROM Profiles GROUP BY ini_cash;" | tr '\n' ' ')

echo ""
echo "✅ 创建完成！"
echo "   - 目标数据库: $TARGET_DB"
echo "   - 用户数量: $USER_COUNT / $NUM_USERS"
echo "   - 策略分布: $STRATEGY_DIST"
echo "   - 资金分布: $CASH_DIST"

if [ "$USER_COUNT" -eq "$NUM_USERS" ]; then
    echo "✅ 验证通过！"
else
    echo "⚠️  警告: 用户数量不匹配！"
fi

