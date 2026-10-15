"""流水线冒烟测试：确保 加载->切分->检索 在轻量模式下跑通。"""
from __future__ import annotations

from rag.config import Config
from rag.loader import load_dir
from rag.pipeline import RAGPipeline


def test_index_and_retrieve():
    cfg = Config(embedding_mode="light", vector_store="numpy")
    docs = load_dir(cfg.data_dir)
    assert docs, "示例语料为空"

    pipe = RAGPipeline(cfg)
    n = pipe.index(docs)
    assert n > 0, "切分片段数应为正"

    res = pipe.ask("什么是余弦相似度？")
    assert res["sources"], "检索应返回来源"
    assert res["sources"][0]["score"] > 0, "最高相似度应为正"
    print("smoke test passed:", n, "chunks, top score =", res["sources"][0]["score"])


if __name__ == "__main__":
    test_index_and_retrieve()
