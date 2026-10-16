"""RAG 评估指标：命中率、Precision@k、MRR。

这些指标帮你量化"换切分长度 / top_k / 重排"对检索质量的影响（M3 里程碑用）。
"""
from __future__ import annotations

from typing import List


def hit_rate(ranked: List[int], relevant: set[int]) -> float:
    """召回结果里是否至少包含一个相关片段。ranked: 片段全局 id 列表。"""
    return 1.0 if any(r in relevant for r in ranked) else 0.0


def precision_at_k(ranked: List[int], relevant: set[int], k: int) -> float:
    """top-k 命中数 / 实际返回条数。

    分母用 len(top) 而非硬编码的 k：当检索结果不足 k 条（召回不足）时，
    按 k 计算会把准确率低估。例如只召回 2 条且都相关，应得 1.0 而非 2/k。
    """
    top = ranked[:k]
    n = len(top)
    if n == 0:
        return 0.0
    return sum(1 for r in top if r in relevant) / n


def mrr(ranked: List[int], relevant: set[int]) -> float:
    """平均倒数排名：第一个相关结果出现位置的倒数。"""
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def evaluate(ranked: List[int], relevant: set[int], k: int = 3) -> dict:
    return {
        "hit_rate": hit_rate(ranked, relevant),
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        "mrr": mrr(ranked, relevant),
    }
