"""检索：向量召回 + 可选关键词重排。"""
from __future__ import annotations
from .embedder import LightEmbedder
from typing import List, Tuple


def _keyword_score(query: str, text: str) -> float:
    """简单关键词重叠率，用于重排（不依赖外部依赖）。

    复用 LightEmbedder._tokens 的中文分词（英文按词、中文按字+二元组），
    保证中文语料下关键词重排依然有效，不再把整段中文当成一个 token。
    """
    q_tokens = LightEmbedder._tokens(query)   # 调用方法，得到 query 的 token 列表
    t_tokens = LightEmbedder._tokens(text)    # 调用方法，得到 text 的 token 列表
    if not q_tokens or not t_tokens:
        return 0.0
    q_set = set(q_tokens)
    hit = sum(1 for w in t_tokens if w in q_set)  # 文本里有多少 token 出现在 query 中
    return hit / len(t_tokens)                      # 重叠率：命中的文本 token 占比


def retrieve(store, query_vec, query_text: str, top_k: int = 3, rerank: bool = True):
    """返回 [(score, meta), ...]，meta 含 text/source/chunk_index。"""
    hits: List[Tuple[float, dict]] = store.search(query_vec, top_k=top_k * (2 if rerank else 1))

    if rerank:
        # 向量分与关键词分加权融合（线性代数① + 概率②的直觉结合）
        reranked = []
        for score, meta in hits:
            kw = _keyword_score(query_text, meta.get("text", ""))
            fused = 0.7 * score + 0.3 * kw
            reranked.append((fused, meta))
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]

    return hits[:top_k]
