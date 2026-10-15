"""向量库：存储片段向量并提供余弦检索。

- NumpyVectorStore：内存暴力检索，纯 Python，轻量/教学用。
- ChromaStore：chromadb 封装，生产上量用（需安装）。
"""
from __future__ import annotations  # 允许注解里使用 list[str] 等新写法（兼容老版本 Python）

import math                  # 数学模块：用来算平方根（向量长度）
from typing import List, Tuple  # 类型注解：List 列表、Tuple 元组


def cosine(a, b) -> float:
    """余弦相似度。a/b 已 L2 归一化时等价于点积。支持 list 或 numpy 数组。"""
    dot = sum(float(x) * float(y) for x, y in zip(a, b))  # 点积：对应位置相乘再求和
    na = math.sqrt(sum(float(x) * float(x) for x in a))   # 向量 a 的长度（L2 范数）
    nb = math.sqrt(sum(float(y) * float(y) for y in b))   # 向量 b 的长度（L2 范数）
    if na == 0 or nb == 0:               # 任一向量长度为 0，无法比较
        return 0.0                       # 直接返回 0（视为不相似）
    return dot / (na * nb)               # 余弦 = 点积 / (a长度 × b长度)


class NumpyVectorStore:
    """内存向量库：把向量和元数据存进列表，查询时暴力算一遍余弦（适合小语料）。"""

    def __init__(self, dim: int):
        self.dim = dim                          # 向量维度（仅作记录，本实现不强制校验）
        self.vectors: List[List[float]] = []    # 已入库的向量列表
        self.meta: List[dict] = []              # 与向量一一对应的元数据（来源、片段号等）

    def add(self, vector, meta: dict) -> None:
        self.vectors.append(list(vector))        # 把一条向量存进列表（转成普通 list 存）
        self.meta.append(meta)                   # 把对应的元数据也存进去

    def add_batch(self, vectors, metas: List[dict], ids=None) -> None:
        for v, m in zip(vectors, metas):         # 遍历批量数据
            self.add(v, m)                       # 逐条调用 add 入库（ids 在轻量模式未使用）

    def search(self, query_vec, top_k: int = 3) -> List[Tuple[float, dict]]:
        if not self.vectors:                     # 库是空的
            return []                            # 直接返回空结果
        sims = [cosine(query_vec, v) for v in self.vectors]  # 对每条库存向量算余弦相似度
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]  # 按相似度从高到低取前 top_k 个下标
        return [(sims[i], self.meta[i]) for i in order]  # 返回 [(相似度, 元数据), ...]


class ChromaStore:
    """chromadb 向量库封装：生产环境用，支持大规模近似检索（属于 20% 留白，需先安装）。"""

    def __init__(self, collection: str = "rag_docs"):
        try:
            import chromadb  # 延迟导入：只有生产模式才需要
        except ImportError as e:  # pragma: no cover
            raise ImportError("生产模式需安装 chromadb：pip install -r requirements.txt") from e
        self._client = chromadb.Client()                          # 创建 chromadb 客户端
        self._col = self._client.get_or_create_collection(collection)  # 获取（或创建）一个集合

    def add_batch(self, vectors, metas: List[dict], ids: List[str]) -> None:
        docs = [m["text"] for m in metas]         # 从元数据里抽出原文，作为可检索文档
        embs = [list(v) for v in vectors]         # 向量转成普通 list
        self._col.add(embeddings=embs, documents=docs, metadatas=metas, ids=ids)  # 一次性写入 chromadb

    def search(self, query_vec, top_k: int = 3) -> List[Tuple[float, dict]]:
        res = self._col.query(query_embeddings=[list(query_vec)], n_results=top_k)  # 用查询向量检索 top_k 条
        out = []
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            out.append((1.0 - float(dist), meta))  # chroma 返回的是"距离"，用 1-距离 转成相似度
        return out


def build_store(kind: str = "numpy", dim: int = 64) -> object:
    if kind == "numpy":                 # 轻量模式：内存暴力检索（默认）
        return NumpyVectorStore(dim)
    if kind == "chromadb":              # 生产模式：chromadb（需先 pip 安装依赖）
        return ChromaStore()
    raise ValueError(f"未知 vector_store: {kind}")  # 非法类型直接报错
