# Git 提交历史记录

> 本文件记录 `sy33456/rag_project` 的提交历史，作为项目开发过程的存档。
> 生成时间：2026-10-17 08:58（GMT+8）

## 仓库信息

| 项 | 值 |
|---|---|
| 仓库地址 | `git@github.com:sy33456/rag_project.git` |
| 分支 | `main`（本地与 origin/main 完全同步） |
| 提交总数 | 5 |
| 跟踪文件数 | 33 |
| 首次提交 | 2026-10-15 |

## 提交历史（从新到旧）

### 1. `2eaa66b` — 2026-10-17 08:55 — sy33456

```
fix(rag): retriever 改用相对导入 from .embedder，删除未使用的 import re
```

- 项目以 `rag` 包方式导入（`app.py` / `main.py` / `pipeline.py` / `run_eval.py` 均 `from rag.xxx`），
  原绝对导入 `from embedder import` 在包上下文会 `ModuleNotFoundError`
- 改为 `from .embedder import LightEmbedder`，与 `pipeline.py` 的 `from . import` 对齐
- 删除未使用的 `import re`
- 验证：`from rag import retriever/pipeline` 正常；`pytest tests/test_pipeline.py tests/test_agent.py` → 8 passed
- 改动：`M rag/retriever.py`

### 2. `d265142` — 2026-10-17 07:19 — sy33456

```
chore: 排除私人笔记(private_notes/)，移除已提交的简历笔记，新增安全推送脚本 git-sync.sh
```

- 3 份私人笔记移入 `private_notes/`（被 `.gitignore` 忽略，仅本地保留）
- 移除此前误提交的 `简历能力对照速查.md`（后经 filter-repo 从全部历史抹除）
- 新增 `git-sync.sh` 安全提交推送脚本
- 改动：`M .gitignore`、`A git-sync.sh`

### 3. `b57e42d` — 2026-10-16 22:12 — sy33456

```
feat: 补充 RAG 检索/评估与文档数据，更新 README 与学习笔记
```

- 新增中文知识文档：`data/llm_basics.txt`、`data/transformer.txt`、`data/vector_db.txt`
- 新增评估脚本 `eval/run_eval.py`
- 更新 `README.md` 与 RAG 学习笔记
- 改动：`M README.md`、`A data/llm_basics.txt`、`A data/transformer.txt`、`A data/vector_db.txt`、`A eval/run_eval.py`

### 4. `92eb2ce` — 2026-10-16 20:13 — sy33456

```
fix: 修复中文关键词重排失效与 precision@k 分母硬编码
```

- `eval/metrics.py`：`precision_at_k` 分母由硬编码 `k` 改为 `len(top)`，召回不足时不再低估精度
- `rag/retriever.py`：`_keyword_score` 修复 `\w+` 无法切分中文的问题，复用 `LightEmbedder._tokens`
  的中文分词（英文按词 / 中文按字 + 二元组），并修复方法未调用、变量未定义的 bug
- 改动：`M eval/metrics.py`、`M rag/retriever.py`
- 注：本提交由两次修复（半截修复 + 补完）经 `git rebase -i` squash 合并而成

### 5. `4f8078e` — 2026-10-15 15:03 — sy33456

```
Initial commit: RAG + Agent 应用落地项目
```

- 初始提交，包含完整 RAG + Agent 项目骨架：
  - `rag/` 核心 11 个模块（agent / chunker / config / embedder / generator / loader /
    logging_setup / pipeline / retriever / vector_store / __init__）
  - 入口 `app.py`（FastAPI）、`main.py`（CLI）、`agent_demo.py`
  - `tests/`（pipeline / agent / app 测试）、`eval/metrics.py`、`data/sample.txt`
  - `Dockerfile`、`.dockerignore`、`.github/workflows/ci.yml`、`requirements.txt`、`.env.example`

## 历史维护记录

本仓库历史经过以下重写操作（均已在推送前备份，并已从 GitHub 远端历史彻底清除目标文件）：

| 操作 | 内容 | 回滚备份 |
|---|---|---|
| filter-repo 清除 #1 | 从全部历史抹除误提交的 `简历能力对照速查.md` | `/tmp/rag_project.pre-purge.bundle` |
| filter-repo 清除 #2 | 从全部历史抹除 3 份学习/面试笔记（学习进度 / 面试准备 / 面试算法刷题清单） | `/tmp/rag_project.pre-study-purge.bundle` |
| rebase squash | 合并两次 retriever 修复提交为一条 | `/tmp/rag_project.pre-squash.bundle` |

> 备份包为一次性临时文件，重启后可能被系统清理。如需长期保留，请另行拷贝。

## 相关说明

- `private_notes/` 目录存放 7 份私人笔记，已被 `.gitignore` 忽略，不会上传
- 日常提交流程：`./git-sync.sh "改动说明"`（只提交指定文件，不触碰 private_notes/）
