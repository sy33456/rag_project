"""检索：向量召回 + 可选关键词重排。"""
from __future__ import annotations

import re
from typing import List, Tuple


def _keyword_score(query: str, text: str) -> float:
    """简单关键词重叠率，用于重排（不依赖外部依赖）。"""
    q = set(re.findall(r"\w+", query.lower()))
    t = re.findall(r"\w+", text.lower())
    if not q or not t:
        return 0.0
    hit = sum(1 for w in t if w in q)
    return hit / len(t)


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
