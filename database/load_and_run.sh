#!/bin/bash

OSS_PATH="/ossfs/workspace/OAG/database"
RAM_PATH="/ramdata"
MONGO_LOG="/ramdata/mongodb.log"

RAM_SIZE="1500G"

if mount | grep "on $RAM_PATH" > /dev/null; then
    echo "✅ 内存盘 $RAM_PATH 已经挂载。"
else
    echo "🛠️ 正在挂载 $RAM_SIZE 内存盘到 $RAM_PATH ..."
    mkdir -p "$RAM_PATH"
    mount -t tmpfs -o size="$RAM_SIZE" tmpfs "$RAM_PATH"
    chmod 777 "$RAM_PATH"
fi

echo "🚀 正在将数据库从 OSS 加载回内存 (IO瓶颈在OSS读取，请耐心等待)..."
if [ -z "$(ls -A $OSS_PATH)" ]; then
    echo "❌ 错误：OSS 路径为空，无法加载！"
    exit 1
fi

rsync -ahP "$OSS_PATH/" "$RAM_PATH/"

echo "✅ 数据加载完成！"

echo "🔥 正在启动 MongoDB..."
mongod --dbpath "$RAM_PATH" \
       --logpath "$MONGO_LOG" \
       --fork \
       --bind_ip_all \
       --nojournal

echo "🎉 数据库已启动！"
echo "你可以通过 python 脚本进行查询了。"
