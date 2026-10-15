"""RAG 流水线：把各模块串起来。"""
from __future__ import annotations

from typing import List, Tuple

from . import chunker, embedder, generator, loader, retriever, vector_store
from .config import Config


class RAGPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.embedder = embedder.build_embedder(
            self.cfg.embedding_mode, self.cfg.embedding_dim, self.cfg.st_model_name
        )
        self.store = vector_store.build_store(self.cfg.vector_store, self.cfg.embedding_dim)
        self.generator = generator.Generator(
            self.cfg.llm_base_url, self.cfg.llm_api_key, self.cfg.llm_model
        )
        self._indexed = False

    def index(self, docs) -> int:
        """docs: [(text, source), ...]。切分 -> 向量化 -> 入库。"""
        chunks = chunker.chunk_docs(docs, self.cfg.chunk_size, self.cfg.chunk_overlap)
        if not chunks:
            return 0
        texts = [c[0] for c in chunks]
        metas = [{"text": t, "source": s, "chunk_index": i} for t, s, i in chunks]
        vectors = self.embedder.embed_batch(texts)
        if hasattr(self.store, "add_batch"):
            ids = [f"{m['source']}-{m['chunk_index']}" for m in metas]
            self.store.add_batch(vectors, metas, ids)
        else:  # numpy
            self.store.add_batch(vectors, metas)
        self._indexed = True
        return len(chunks)

    def ask(self, question: str) -> dict:
        if not self._indexed:
            raise RuntimeError("请先调用 index() 建立索引")
        q_vec = self.embedder.embed(question)
        hits = retriever.retrieve(
            self.store, q_vec, question, self.cfg.top_k, self.cfg.rerank
        )
        answer = self.generator.generate(question, hits)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"score": round(s, 4), "source": m.get("source"), "chunk_index": m.get("chunk_index")}
                for s, m in hits
            ],
        }
