"""递归字符切分：长文本 -> 带重叠的片段列表。"""
from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """按字符窗口切分，步长 = chunk_size - overlap。

    要点：重叠窗口保留跨边界的语义，避免把一个完整句子劈成两半。
    """
    text = text.strip()
    if not text:
        return []

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def chunk_docs(docs, chunk_size: int = 500, overlap: int = 50):
    """docs: [(text, source), ...] -> [(chunk_text, source, chunk_index), ...]"""
    out = []
    for text, source in docs:
        for i, c in enumerate(chunk_text(text, chunk_size, overlap)):
            out.append((c, source, i))
    return out
