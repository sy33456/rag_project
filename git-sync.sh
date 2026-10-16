#!/bin/bash
# git-sync.sh —— 改完代码后，一条命令安全提交并推送到 GitHub
#
# 用法：
#   ./git-sync.sh "你的提交说明"
#   ./git-sync.sh "提交说明" -y      # 跳过确认直接推送
#
# 安全设计：
#   - 私人笔记（面试讲稿/简历/代码合集）已在 .gitignore 中排除，不会被提交
#   - 提交前会用 git status 让你看清「哪些文件会被改」，并二次确认
#   - 远程跟踪已建好，直接 git push 即可，无需 -u origin main

set -e
cd "$(dirname "$0")"

# 1) 必须有提交说明
if [ -z "$1" ]; then
  echo "❌ 用法: ./git-sync.sh \"提交说明\""
  exit 1
fi
MSG="$1"
AUTO="no"
[ "$2" = "-y" ] && AUTO="yes"

# 2) 先看清楚当前状态（这就是「读 git status」的核心）
echo "==================== git status ===================="
git status --short
echo "===================================================="

# 3) 暂存所有改动（私人笔记因 .gitignore 已被排除，安全）
git add -A

echo "----- 本次将提交的文件 -----"
git diff --cached --name-only
if [ -z "$(git diff --cached --name-only)" ]; then
  echo "（没有可提交的改动，已退出）"
  exit 0
fi

# 4) 二次确认，避免误推
if [ "$AUTO" = "no" ]; then
  read -p "确认提交并推送到 GitHub? [y/N] " ans
  case "$ans" in
    y|Y) ;;
    *) echo "已取消，撤销暂存"; git reset -q; exit 0;;
  esac
fi

# 5) 提交 + 推送
git commit -m "$MSG"
git push
echo "✅ 已推送到 GitHub: sy33456/rag_project"
