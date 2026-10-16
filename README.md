# rag_project —— 可运行的 RAG 文档问答骨架

一个为**应用层 AI 工程师**设计的检索增强生成（RAG）最小骨架。
两条运行路径：

- **轻量模式（默认，零外部依赖）**：用 numpy 做哈希词袋 embedding + 余弦检索，开箱即跑，用于理解整条流水线。
- **生产模式（可选）**：切换为 `sentence-transformers` 真实 embedding + `chromadb` 向量库 + OpenAI 兼容大模型，结构完全一致。

---

## 系统架构

```
用户问题
   │
   ▼
[ retriever ]  ── 问题向量化 → 向量库余弦检索 top_k → (可选)重排
   │  命中片段
   ▼
[ generator ]  ── 拼接 Prompt(上下文+问题) → 调用 LLM → 带引用回答
   │
   ▼
回答 + 引用来源
```

模块职责（见 `rag/` 目录）：

| 文件 | 职责 |
|------|------|
| `config.py` | 全局配置（切分长度、top_k、embedding 模式、LLM 地址）|
| `loader.py` | 文档加载（.txt/.md），可扩展 PDF/网页 |
| `chunker.py` | 递归字符切分 + 重叠窗口 |
| `embedder.py` | 向量化：轻量哈希词袋 / 真实句向量（可切换）|
| `vector_store.py` | 向量库：numpy 余弦 / chromadb（可切换）|
| `retriever.py` | 检索 + 关键词重排 |
| `generator.py` | LLM 调用封装（OpenAI 兼容）+ 无 key 时的模板兜底 |
| `pipeline.py` | 串起 加载→切分→向量化→检索→生成 |
| `agent.py` | **Agent 编排**：把 `/query`、`/ingest`、`/health` 封装为可调用工具，支持 demo 意图路由与 LLM Function Calling 两种模式 |

服务与工具：

| 文件 | 职责 |
|------|------|
| `main.py` | CLI 演示：建索引 → 问答 |
| `app.py` | FastAPI 服务：`POST /query`、`POST /ingest`、`GET /health`，带缓存与请求日志 |
| `rag/logging_setup.py` | 统一日志配置（级别由 `LOG_LEVEL` 控制） |
| `eval/metrics.py` | 评估指标：命中率、Precision@k、MRR |
| `eval/run_eval.py` | 评估脚本：12 条标注测试集一键跑出指标（当前实测 top-3 命中率 92% / MRR 0.78） |
| `tests/test_pipeline.py` | 流水线冒烟测试 |
| `tests/test_app.py` | API 测试（未装 fastapi 时自动跳过） |
| `tests/test_agent.py` | Agent 测试：工具定义、demo 意图路由、calculator 安全、LLM 函数调用循环 |
| `agent_demo.py` | Agent 演示：展示"意图→选工具→执行"全链路（服务在线则真实调用，否则 Mock） |
| `data/*.txt` | 示例语料：sample（RAG 入门）/ transformer / vector_db / llm_basics |

工程化（M5）：`Dockerfile`（slim 镜像 + 层缓存 + 非 root 用户）、`.dockerignore`、`.github/workflows/ci.yml`（自动测试 + 镜像构建校验）。

---

## 技术选型说明

| 环节 | 轻量模式 | 生产模式 | 选型理由 |
|------|----------|----------|----------|
| Embedding | 哈希词袋（numpy） | `sentence-transformers` | 轻量模式零依赖可教学；生产用本地句向量，免 API 费用、可离线 |
| 向量库 | numpy 暴力检索 | `chromadb` | 小规模 demo 暴力检索足够；生产上量后用 ANN 向量库 |
| 生成 | 模板兜底 | OpenAI 兼容 API（GPT / 本地 Ollama / 通义等）| 统一接口，换模型只改配置 |
| 服务 | — | `fastapi` + `uvicorn` | 异步、自动文档，应用层主流 |

---

## 快速开始

```bash
cd rag_project

# 1) 轻量模式（无需安装任何依赖，只要 Python 3.10+）
python main.py

# 2) 生产模式
pip install -r requirements.txt
cp .env.example .env      # 填入 LLM_API_KEY
python main.py --mode prod

# 3) 启动服务（本地）
uvicorn app:app --reload  # 访问 http://127.0.0.1:8000/docs

# 4) 启动服务（Docker）
docker build -t rag_project .
docker run -p 8000:8000 --env-file .env rag_project
```

服务端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/query` | 问答，返回回答 + 引用来源；相同问题 300s 内命中缓存 |
| POST | `/ingest` | 重新加载 `data/` 并重建索引（语料更新后无需重启） |
| GET | `/health` | 健康检查，返回已索引片段数 |

---

## 里程碑（从骨架到能写进简历的项目）

1. **M1 跑通流水线**：轻量模式完成 加载→切分→检索→生成（本骨架已具备）。
2. **M2 换真实向量**：接入 `sentence-transformers` + `chromadb`，对比检索质量。
3. **M3 加评估**：用 `eval/metrics.py` + `eval/run_eval.py` 量化 top_k / 重排 / chunk_size 的影响（当前示例语料实测 top-3 命中率 92%、MRR 0.78，`python eval/run_eval.py` 可复现）。
4. **M4 接 LLM 与引用**：真实生成 + 返回来源片段，做"可溯源"问答。
5. **M5 工程化**（已完成）：FastAPI 服务化（`/query` `/ingest` `/health`）、批量入库、TTL 缓存、结构化日志、Docker、CI。
6. **M6 Agent 编排**（已完成）：新增 `agent.py`，把 RAG 服务接口封装为 Tool Use / Function Calling 工具，Agent 自主决定调用检索/入库/健康检查，并支持组合外部工具（计算器 demo）；含 `agent_demo.py` 与 `tests/test_agent.py`。
7. **M7 进阶**：混合检索（BM25+向量）、重排模型、多轮对话记忆、流式输出。

> 数学落点：检索靠**余弦相似度**（线性代数①），采样/温度靠**概率**（②），训练直觉靠**梯度下降**（③），性能靠**大 O**（④）。建议边做 M1–M3 边回看 `math_cheatsheet.md`。
