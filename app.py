"""FastAPI 服务：RAG 问答 API。

端点：
  POST /query   问答（带引用来源）；命中缓存直接返回，省一次 LLM 调用
  POST /ingest  重新加载 data_dir 并重建索引（语料更新后无需重启服务）
  GET  /health  健康检查

启动：uvicorn app:app --reload  ->  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import time

from pydantic import BaseModel

from rag import vector_store
from rag.config import Config
from rag.loader import load_dir
from rag.logging_setup import setup_logging
from rag.pipeline import RAGPipeline

# 为什么在入口统一初始化：服务只启动一次，各模块不再重复配置
setup_logging()
logger = logging.getLogger("rag.app")

cfg = Config.from_env()
pipe = RAGPipeline(cfg)
n = pipe.index(load_dir(cfg.data_dir))
logger.info("服务启动：已索引 %d 个片段（模式=%s）", n, cfg.embedding_mode)

# ---- 查询缓存 ----
# 为什么要缓存：LLM 调用按 token 计费且慢，相同问题直接复用上次结果
CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "300"))  # 秒
_cache: dict[str, tuple[float, dict]] = {}


def _cache_get(question: str) -> dict | None:
    hit = _cache.get(question)
    if hit is not None and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    _cache.pop(question, None)  # 顺手清掉过期项，防止字典无限增长
    return None


def _cache_set(question: str, res: dict) -> None:
    _cache[question] = (time.time(), res)


class QueryReq(BaseModel):
    question: str


class SourceItem(BaseModel):
    score: float
    source: str | None = None
    chunk_index: int | None = None


class QueryResp(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]


class IngestResp(BaseModel):
    indexed_chunks: int


def create_app():
    from fastapi import FastAPI, Request

    app = FastAPI(title="RAG 文档问答 API")
    app.state.pipe = pipe
    app.state.chunk_count = n  # ingest 后更新它，health 才反映真实状态

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # 为什么记请求日志：出问题时第一现场是"哪个请求、耗时多久、什么状态码"
        start = time.perf_counter()
        response = await call_next(request)
        cost_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method, request.url.path, response.status_code, cost_ms,
        )
        return response

    @app.post("/query", response_model=QueryResp)
    def query(req: QueryReq) -> QueryResp:
        cached = _cache_get(req.question)
        if cached is not None:
            logger.info("缓存命中：%s", req.question[:30])
            return QueryResp(**cached)
        res = app.state.pipe.ask(req.question)
        _cache_set(req.question, res)
        return QueryResp(
            question=res["question"],
            answer=res["answer"],
            sources=[SourceItem(**s) for s in res["sources"]],
        )

    @app.post("/ingest", response_model=IngestResp)
    def ingest() -> IngestResp:
        # 为什么重建而不是追加：向量库的 add_batch 是追加语义，
        # 直接再插一遍会得到重复片段；换一个新 store 最干净。
        app.state.pipe.store = vector_store.build_store(
            cfg.vector_store, cfg.embedding_dim
        )
        _cache.clear()  # 索引变了，旧缓存全部作废
        cnt = app.state.pipe.index(load_dir(cfg.data_dir))
        app.state.chunk_count = cnt
        logger.info("重新入库完成：%d 个片段", cnt)
        return IngestResp(indexed_chunks=cnt)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "indexed_chunks": app.state.chunk_count}

    return app


app = create_app()
