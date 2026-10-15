# 推送到 GitHub 指南

## 现状
- 本地仓库已在 `rag_project/` 初始化并提交（Initial commit）。
- **但 push 需要 GitHub 登录认证**，你目前登不进官网，所以这一步要你本人完成。
- 我已把 `.env`、`.venv`、`__pycache__` 等排除，密钥不会泄露。

---

## 第一步：恢复 GitHub 登录（用你记得的邮箱）
1. 打开 https://github.com/password_reset （或官网登录页点 "Forgot password?"）
2. 输入你记得的邮箱 → GitHub 发重置链接到该邮箱
3. 去邮箱收信 → 点链接设新密码 → 登录
4. 如果开了 2FA 且收不到验证码：走账号恢复
   https://github.com/account/recovery ，提交身份证明，GitHub 支持会帮你解锁（可能要几天）

## 第二步：创建远程仓库
- 登录后右上角 `+` → New repository
- 名字建议：`rag-agent-app`
- 选 **Public**（作品集要公开，面试官才能看）
- **不要**勾 "Add a README"（我们已有 README.md）
- 其他默认 → Create repository
- 创建后复制仓库 URL（HTTPS 或 SSH 都行）

## 第三步：本地关联并推送
把 URL 换成你刚复制的：
```bash
cd /Users/sheyu/WorkBuddy/2026-09-25-20-02-45/rag_project
git remote add origin <你的仓库URL>
git branch -M main
git push -u origin main
```
- **HTTPS 方式**：弹窗要用户名 + 密码；密码处填 **Personal Access Token**（不是账号密码）。
  Token 生成：GitHub → Settings → Developer settings → Personal access tokens → 勾 `repo` 权限。
- **SSH 方式**：需先把本地公钥 `~/.ssh/id_ed25519.pub` 内容加到 GitHub → Settings → SSH keys。

## 第四步：改成你自己的 git 身份（重要）
当前初始 commit 用的是占位身份，请改成你的（否则 GitHub 不会把贡献算到你账号）：
```bash
git config user.name "你的名字"
git config user.email "你记得的那个邮箱"
git commit --amend --reset-author   # 改写初始 commit 的作者信息
```

## 第五步：把链接填进简历
仓库建好并 push 后，把 URL 填到简历「作品集 / GitHub」处（简历里留了【】占位）。

---

## 可选：让我帮你 push
等你恢复登录后，如果愿意提供 **Personal Access Token**（别给账号密码），我可以用 git 命令直接帮你 push；
或者你本地装了 GitHub CLI 后 `gh auth login`，我也能协助后续操作。
