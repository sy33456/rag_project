# RAG 项目 · 整合学习笔记（读到 简历 80% 专用）

> 你目前只学过 **pytest + numpy 基础**。这份笔记把 `rag_project` 的全部代码、讲解、动手实验整合在一起，  
> 你**不需要再打开任何源文件**，从头读到尾就能把简历上写的 RAG 项目讲到面试水平（约 80%）。  
> 代码全部来自项目真实文件，行号标注方便你回头对照。**每行代码后都加了中文注释**。

---

## 〇、目标与范围（先看清 80% 是什么）

**要能讲、能演示（核心 80%）：**

- RAG 全流程概念 + `pipeline` 串讲
- 分块策略（重叠窗口）及**为什么重叠**
- Embedding 向量化概念（轻量模式原理）
- 余弦相似度（用 numpy 算）
- 检索 + 关键词重排（0.7 / 0.3）
- 大模型 API 调用 + 无 Key 兜底
- FastAPI 三接口 + 查询缓存 + `/ingest` 重建逻辑
- 效果评估：`hit_rate` / `precision@k` / `mrr`
- 跑通项目 + 写/跑测试
- Docker 部署思路

**诚实留白（占 20%，面试说"原理了解、未实践"）：**

- 生产模式真实句向量（sentence-transformers）+ chromadb 向量库 → 未安装
- 模型微调（fine-tuning）→ 没做过
- 前端交互 / 分布式系统设计

---

## 一、学习路线总览（8 课）

| 课次 | 主题           | 你学完能做到                 |
| -- | ------------ | ---------------------- |
| S0 | 跑通 + 直觉      | 一句话说清 RAG，看到它运行        |
| S1 | 向量与余弦（numpy） | 用 numpy 算余弦；懂"语义近→向量近" |
| S2 | 分块           | 讲清重叠窗口为什么必要            |
| S3 | 检索与重排        | 讲清为什么检索后还要重排           |
| S4 | 生成与流水线       | 讲清"检索→拼 Prompt→调模型"    |
| S5 | 效果评估         | 定义三个指标并说明用途            |
| S6 | 服务与部署        | 讲清三接口、缓存、重建索引          |
| S7 | 测试 + 自己动手    | 跑 pytest；写第一处自己的代码     |

---

## S0 · 跑通与建立直觉

**先跑起来（零依赖，直接复制）：**

```bash
cd /Users/sheyu/WorkBuddy/2026-09-25-20-02-45/rag_project   # 进入项目目录
/Users/sheyu/.workbuddy/binaries/python/envs/default/bin/python main.py  # 用工作区自带的 Python 跑入口脚本
```

出现 `>>` 后输入问题（如"什么是余弦相似度？"），回车看回答；输入空行或 `Ctrl+C` 退出。  
不配 `LLM_API_KEY` 时是"演示模式"，直接把检索到的原文贴回来。

**`main.py` 在做什么（57 行白话）：**

- `load_dir(data_dir)`：把 `data/` 里的 `.txt/.md` 读成 `(文本, 来源)` 列表 → **第1步 加载文档**
- `RAGPipeline(cfg)` + `pipe.index(docs)`：切分→向量化→入库 → **第2步 切分 + 第3步 向量化**
- `pipe.ask(q)`：每次提问时检索最相关片段并生成答案 → **第4步 检索与生成**
- 打印 `answer` + `sources`（含相似度分数）

**一句话记住 RAG：**

> 先从一个知识库里检索相关片段，拼进 Prompt 让大模型基于资料作答，减少幻觉、答案可追溯。

**你会看到的输出里**：`sample.txt#1 相似度=0.36` —— 这就是 `ask()` 给每段内容打的"语义接近分"。

---

## S1 · 向量与余弦（用你会的 numpy）

**为什么文本能变成数字？** Embedding 就是把一段文字映射成一串固定长度的数字（向量）。语义相近的文字，向量"方向"也相近。

### 1.1 项目里文本怎么变成向量 —— `rag/embedder.py`

```python
class LightEmbedder:                                   # 轻量向量化器类：把文本变成向量
    def __init__(self, dim: int = 64):                 # 初始化；dim = 向量维度，默认 64
        self.dim = dim                                 # 把维度存到对象，后面 embed 会用

    @staticmethod                                     # 静态方法：不依赖对象，只做"文本→词列表"
    def _tokens(text: str) -> List[str]:               # 输入文本，输出"词"的列表
        tokens = re.findall(r"[a-zA-Z]+", text.lower())  # 英文按单词提取并转小写
        han = re.findall(r"[一-鿿]", text)               # 中文按单字提取（中文无空格）
        tokens.extend(han)                             # 先把中文单字加入词列表
        tokens.extend(han[i] + han[i + 1] for i in range(len(han) - 1))  # 再加"相邻两字"组合，增强中文重叠信号
        return tokens                                  # 返回最终的词列表

    def embed(self, text: str) -> List[float]:         # 把一段文本变成一串小数（向量）
        vec = [0.0] * self.dim                         # 造一个长度=dim、全 0 的列表，作为向量
        for tok in self._tokens(text):                 # 遍历分出来的每个"词"
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)  # 把词 md5 哈希成一个大整数
            vec[h % self.dim] += 1.0                    # 用"哈希 mod 维度"定位桶，该桶计数 +1（词袋+哈希技巧）
        norm = math.sqrt(sum(v * v for v in vec))      # 算向量 L2 长度（各分量平方和再开方）
        if norm > 0:                                   # 防止除以 0
            vec = [v / norm for v in vec]               # 归一化：每分量除以长度，使向量变成"单位长度"
        return vec                                     # 返回归一化后的向量
```

**白话**：把每个词哈希到一个 64 格的桶里计数，再除以长度归一化。它是**教学用的近似 embedding**（确定性、零依赖），不是真正的语义向量——真正语义向量要用 `STEmbedder`（sentence-transformers），那个属于 20% 留白。

### 1.2 余弦相似度 —— `rag/vector_store.py`

```python
def cosine(a, b) -> float:                            # 计算两个向量的余弦相似度
    dot = sum(float(x) * float(y) for x, y in zip(a, b))  # 点积：对应位置相乘再求和
    na = math.sqrt(sum(float(x) * float(x) for x in a))   # 向量 a 的长度（L2 范数）
    nb = math.sqrt(sum(float(y) * float(y) for y in b))   # 向量 b 的长度（L2 范数）
    if na == 0 or nb == 0:                              # 任一向量长度为 0，无法比较
        return 0.0                                       # 直接返回 0（视为不相似）
    return dot / (na * nb)                              # 余弦 = 点积 / (a长度 × b长度)
```

**白话**：余弦只比"方向"不比"长度"。值越接近 1，两向量越同向（语义越像）；接近 0 表示无关；可为负表示反向。

### 1.3 用 numpy 亲手验证（动手试）

```python
import numpy as np                                   # 导入 numpy 并起别名 np
a = np.array([1.0, 2.0, 3.0])                        # 造向量 a
b = np.array([2.0, 4.0, 6.0])                         # 造向量 b（是 a 的 2 倍，方向相同）
cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # numpy 版余弦：点积 / 两长度之积
print(cos)   # 1.0（b 是 a 的倍数，方向完全相同）
```

跑出 1.0，你就验证了"余弦 = 点积 ÷ 两向量长度之积"，和项目 `cosine()` 是同一件事。

**面试能讲**：文本→向量（embedding）；相似度用余弦，只看方向；向量已归一化时余弦等价于点积。

---

## S2 · 分块策略 —— `rag/chunker.py`

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:  # 把长文本切成小片段
    text = text.strip()                               # 去掉首尾空白，避免空片段
    if not text:                                      # 文本为空
        return []                                     # 直接返回空列表
    step = max(1, chunk_size - overlap)                # 滑动步长 = 窗口 - 重叠（如 500-50=450）
    chunks = []                                       # 用来装切出来的片段
    start = 0                                         # 当前片段起点
    while start < len(text):                           # 还没切到文本末尾就继续
        end = min(start + chunk_size, len(text))       # 片段终点（不超过文本总长）
        chunk = text[start:end].strip()                # 截取这一段并去空白
        if chunk:                                      # 若这段非空
            chunks.append(chunk)                       # 加入结果列表
        if end == len(text):                           # 已经到文本末尾
            break                                      # 结束循环
        start += step                                  # 起点往前滑一个步长（= 窗口 - 重叠）
    return chunks                                      # 返回所有片段
```

**为什么带重叠（面试必问）：** 如果刚好在句子中间切断，语义就断了。重叠让边界处的句子被两个片段都覆盖，检索时不容易漏。  
**参数**：`chunk_size=500`（每片最多 500 字符）、`overlap=50`（相邻片共享 50 字符），`step=450` 就是每次往前滑 450。

---

## S3 · 检索与重排 —— `rag/retriever.py`

```python
def _keyword_score(query: str, text: str) -> float:   # 算"文本"里命中了多少查询词（比例）
    q = set(re.findall(r"\w+", query.lower()))         # 把查询拆成词集合（去重、小写）
    t = re.findall(r"\w+", text.lower())               # 把候选文本拆成词列表
    if not q or not t:                                 # 查询或文本没有词
        return 0.0                                     # 无法比较，返回 0
    hit = sum(1 for w in t if w in q)                   # 统计文本中有多少个词在查询里出现
    return hit / len(t)                                # 命中词数 / 文本总词数 = 关键词重叠率


def retrieve(store, query_vec, query_text, top_k=3, rerank=True):  # 从向量库检索并（可选）重排
    hits = store.search(query_vec, top_k=top_k * (2 if rerank else 1))   # 先多召回一倍候选（重排时才加倍）
    if rerank:                                         # 需要做重排
        reranked = []                                  # 放融合后的结果
        for score, meta in hits:                       # 遍历每个候选（向量相似度分 + 元数据）
            kw = _keyword_score(query_text, meta.get("text", ""))  # 算这个候选和查询的关键词重叠率
            fused = 0.7 * score + 0.3 * kw             # 融合：向量分占 0.7，关键词分占 0.3
            reranked.append((fused, meta))             # 把融合分和元数据一起存起来
        reranked.sort(key=lambda x: x[0], reverse=True)  # 按融合分从高到低排序
        return reranked[:top_k]                        # 只取前 top_k 个
    return hits[:top_k]                                # 不重排就直接取前 top_k
```

**为什么检索后还要重排（面试必问）：** 纯向量检索对精确词（型号、编号、人名）不敏感。先用向量召回 top_k*2 个候选，再用关键词重叠率补一刀，两者 **0.7 / 0.3 加权融合**后取前 k，更稳。

---

## S4 · 生成与流水线

### 4.1 生成 —— `rag/generator.py`

```python
PROMPT_TEMPLATE = """你是一个严谨的问答助手。只根据下面的【上下文】回答，若上下文不足以回答请说"资料中未提及"。
【上下文】
{context}
【问题】
{question}
【回答】"""                                                # 提示词模板：强制模型"只根据上下文答"

class Generator:
    def generate(self, question, contexts):             # 根据问题 + 检索到的上下文生成答案
        ctx_text = "\n\n".join(                          # 把多个片段拼成一段上下文文本
            f"[来源 {m.get('source')}#{m.get('chunk_index')}] {m.get('text')}" for _, m in contexts  # 每条带上"来源#片段号"
        )
        prompt = PROMPT_TEMPLATE.format(context=ctx_text, question=question)  # 把上下文和问题填进模板
        if self.api_key:                                 # 如果配了 LLM 密钥
            try:
                return self._call_llm(prompt)             # 调用 OpenAI 兼容接口拿答案
            except Exception as e:                        # 调用失败（网络/额度等）
                return f"（LLM 调用失败：{e}）\n\n以下是检索到的上下文：\n{ctx_text}"  # 失败也返回上下文，便于排查
        return f"（演示模式：未配置 LLM_API_KEY，以下为检索到的上下文）\n\n{ctx_text}"  # 无 Key 兜底：直接返回检索到的原文
```

**白话**：RAG 核心 = 把检索到的上下文塞进 Prompt，让模型"只根据资料答"。没配 Key 就直接把上下文返回，方便看流水线跑通。

### 4.2 流水线串讲 —— `rag/pipeline.py`

```python
def index(self, docs):                                  # 建索引：把语料切分→向量化→入库
    chunks = chunker.chunk_docs(docs, self.cfg.chunk_size, self.cfg.chunk_overlap)  # 调用 chunker 切分
    texts = [c[0] for c in chunks]                       # 取出每个片段的纯文本
    metas = [{"text": t, "source": s, "chunk_index": i} for t, s, i in chunks]  # 构造元数据（文本/来源/片段号）
    vectors = self.embedder.embed_batch(texts)           # 把所有片段批量向量化
    self.store.add_batch(vectors, metas, ids)            # 向量 + 元数据一起入库
    self._indexed = True                                # 标记"已建索引"
    return len(chunks)                                   # 返回切出了多少片段

def ask(self, question):                                # 问答：用户提问时执行
    q_vec = self.embedder.embed(question)                # 先把问题也向量化
    hits = retriever.retrieve(self.store, q_vec, question, self.cfg.top_k, self.cfg.rerank)  # 检索 + 重排
    answer = self.generator.generate(question, hits)     # 把上下文拼进 Prompt 调大模型（或兜底）
    return {"question": question, "answer": answer,      # 返回问题、答案
            "sources": [{"score": round(s,4), "source": m.get("source"),  # 以及来源列表（相似度保留4位小数）
                         "chunk_index": m.get("chunk_index")} for s, m in hits]}
```

**面试能讲全链路**：文档→切分→向量化→向量检索+重排→拼 Prompt 调大模型→带引用回答。

---

## S5 · 效果评估 —— `rag/eval/metrics.py`

```python
def hit_rate(ranked, relevant):                         # 召回结果里是否至少含一个相关片段
    return 1.0 if any(r in relevant for r in ranked) else 0.0  # 有任意一个相关 → 1.0，否则 0.0

def precision_at_k(ranked, relevant, k):                # 前 k 个结果里相关的占比
    top = ranked[:k]                                    # 取前 k 个
    return sum(1 for r in top if r in relevant) / k if k else 0.0  # 相关数 / k（k=0 时返回 0）

def mrr(ranked, relevant):                              # 第一个相关结果出现位置的倒数
    for i, r in enumerate(ranked, start=1):              # 从第 1 位开始数
        if r in relevant:                               # 找到第一个相关的
            return 1.0 / i                               # 返回 1/位置（越靠前分越高）
    return 0.0                                          # 一个都没找到 → 0
```

**三个指标干嘛用（面试能讲）：**

- `hit_rate`：有没有至少召回一个对的（粗粒度"找没找到"）。
- `precision@k`：前 k 个结果里，有多大比例是相关的（精不精）。
- `mrr`：第一个相关结果排第几，越靠前分越高（排得好不好）。

**价值**：改 `chunk_size`、改 `top_k`、开关重排，看这三个数怎么变，就知道调参有没有用，而不是靠感觉。

---

## S6 · 服务与部署 —— `rag_project/app.py` + `Dockerfile`

**三接口（`create_app`，行 71-121）：**

- `POST /query`：问答（带引用来源）；命中缓存直接返回，省一次 LLM 调用。
- `POST /ingest`：重建索引（语料更新后不用重启服务）。
- `GET /health`：健康检查，返回已索引片段数。

**查询缓存（为什么）：** LLM 按 token 计费且慢，相同问题直接复用结果（`CACHE_TTL=300` 秒）。

**`/ingest` 为什么重建而不是追加（面试能讲）：** 向量库的 `add_batch` 是**追加**语义，再插一遍会得到重复片段；换个新 store 最干净。这也是测试防回归的点（重建后数量不翻倍）。

**日志中间件：** 记录每个请求的方法、路径、状态码、耗时（出问题时第一现场）。

**Dockerfile 思路（能讲即可）：** 用 slim 基础镜像、非 root 用户运行、暴露端口——保证"在我机器能跑"变成"在哪都能跑"。

---

## S7 · 测试与亲手写第一处代码

**现有测试范例** `rag_project/tests/test_app.py`：测了 `health`、`/query` 带来源、`test_query_cache_hit`（相同问题两次结果一致）、`test_ingest_rebuilds_index`（重建后数量不翻倍）。

**动手任务（二选一，做完项目就真正有"你写的部分"）：**

*任务 A：给 `main.ask()` 写个测试*

```python
# tests/test_main.py
from rag.config import Config        # 导入配置类（集中存放切分/检索等参数）
from rag.loader import load_dir      # 导入语料加载函数
from rag.pipeline import RAGPipeline # 导入 RAG 流水线类

def test_ask_returns_sources():      # pytest 会自动发现并运行以 test_ 开头的函数
    cfg = Config()                   # 用默认配置（轻量模式）
    pipe = RAGPipeline(cfg)          # 建流水线
    pipe.index(load_dir(cfg.data_dir))  # 把 data/ 下的语料建索引
    res = pipe.ask("什么是余弦相似度？")  # 问一个问题
    assert res["answer"]             # 断言：答案非空
    assert res["sources"]            # 断言：带来了源（说明真的检索到了）
```

*任务 B：给 `app.py` 加一个 `GET /stats` 接口*（在 `create_app` 内、`return app` 之前加）

```python
@app.get("/stats")                   # 注册一个 GET 接口，路径 /stats
def stats() -> dict:                 # 返回字典，FastAPI 会自动转成 JSON
    return {"indexed_chunks": app.state.chunk_count,   # 已索引的片段数
            "embedding_mode": cfg.embedding_mode,       # 当前向量化模式（light/st）
            "vector_store": cfg.vector_store}           # 当前向量库类型（numpy/chromadb）
```

加完跑 `pytest` 确认没破坏现有测试。这处就是你"自己写的"，面试底气不同。

---

## 达标自测（闭眼能答 = 80% 到手）

1. 为什么分块要带重叠？
2. 为什么检索后还要重排？
3. 余弦相似度是什么、怎么算？
4. `hit_rate` / `precision@k` / `mrr` 分别是干嘛的？
5. `/ingest` 为什么要重建索引而不是追加？

---

## 拓展阅读（云端资料库，按需打开）

- **Python 资料库** `x50KE4XExZE54oxAHWwlOr`：Part 4 numpy 进阶（向量/点积/范数，S1 数学底子）+ 文末「附录·专业词汇通俗词典」（embedding/向量 通俗解释）。
- **LLM 资料库** `EAHg8O3c8XLjVlVfOpELjE`：RAG / Embedding / 大模型应用章节（S1、S4 概念背景）。
- **本地配套**：`numpy_pandas_guide.md`（numpy 基础）、`math_cheatsheet.md`（余弦相似度公式→AI 场景）。

> 需要我打开云端文档并贴具体段落，告诉我即可。
