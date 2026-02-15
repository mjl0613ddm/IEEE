#!/bin/bash

# 检查数据库一致性：确保所有用户都有对应的策略记录

DB_PATH="${1:-data/sys_100.db}"

if [ ! -f "$DB_PATH" ]; then
    echo "❌ 数据库文件不存在: $DB_PATH"
    echo "用法: $0 [数据库路径]"
    echo "示例: $0 data/sys_100.db"
    exit 1
fi

echo "=== 检查数据库一致性: $DB_PATH ==="
echo ""

# 检查Profiles表中的用户数量
PROFILES_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(DISTINCT user_id) FROM Profiles;" 2>/dev/null)
echo "📊 Profiles表中的用户数量: $PROFILES_COUNT"

# 检查Strategy表中的用户数量
STRATEGY_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(DISTINCT user_id) FROM Strategy;" 2>/dev/null)
echo "📊 Strategy表中的用户数量: $STRATEGY_COUNT"

echo ""

# 找出在Profiles中但不在Strategy中的用户
echo "🔍 检查缺失的策略记录..."
MISSING_USERS=$(sqlite3 "$DB_PATH" "
    SELECT p.user_id 
    FROM Profiles p 
    LEFT JOIN Strategy s ON p.user_id = s.user_id 
    WHERE s.user_id IS NULL 
    LIMIT 20;
" 2>/dev/null)

if [ -z "$MISSING_USERS" ]; then
    echo "✅ 所有用户都有对应的策略记录！"
else
    MISSING_COUNT=$(sqlite3 "$DB_PATH" "
        SELECT COUNT(*) 
        FROM Profiles p 
        LEFT JOIN Strategy s ON p.user_id = s.user_id 
        WHERE s.user_id IS NULL;
    " 2>/dev/null)
    echo "❌ 发现 $MISSING_COUNT 个用户缺少策略记录："
    echo "$MISSING_USERS" | while read user_id; do
        echo "   - $user_id"
    done
fi

echo ""

# 找出在Strategy中但不在Profiles中的用户（孤立记录）
echo "🔍 检查孤立的策略记录..."
ORPHAN_USERS=$(sqlite3 "$DB_PATH" "
    SELECT s.user_id 
    FROM Strategy s 
    LEFT JOIN Profiles p ON s.user_id = p.user_id 
    WHERE p.user_id IS NULL 
    LIMIT 20;
" 2>/dev/null)

if [ -z "$ORPHAN_USERS" ]; then
    echo "✅ 没有孤立的策略记录！"
else
    ORPHAN_COUNT=$(sqlite3 "$DB_PATH" "
        SELECT COUNT(*) 
        FROM Strategy s 
        LEFT JOIN Profiles p ON s.user_id = p.user_id 
        WHERE p.user_id IS NULL;
    " 2>/dev/null)
    echo "⚠️  发现 $ORPHAN_COUNT 个孤立的策略记录（在Strategy中但不在Profiles中）："
    echo "$ORPHAN_USERS" | while read user_id; do
        echo "   - $user_id"
    done
fi

echo ""

# 检查策略分布
echo "📈 策略分布统计："
sqlite3 "$DB_PATH" "
    SELECT strategy, COUNT(*) as count 
    FROM Strategy 
    GROUP BY strategy;
" 2>/dev/null | while IFS='|' read strategy count; do
    echo "   - $strategy: $count 个用户"
done

echo ""

# 总结
if [ -z "$MISSING_USERS" ] && [ -z "$ORPHAN_USERS" ]; then
    echo "✅ 数据库一致性检查通过！"
    exit 0
else
    echo "❌ 数据库存在一致性问题，请修复后再运行模拟。"
    exit 1
fi

