"""向量化：把文本变成向量。

- LightEmbedder：哈希词袋（纯 Python，零依赖），用于教学/演示。
- STEmbedder：sentence-transformers 真实句向量，生产用（需安装）。
"""
from __future__ import annotations  # 允许注解里使用 list[str] 等新写法（兼容老版本 Python）

import hashlib               # 哈希模块：把"词"映射成一个固定整数（哈希值）
import math                  # 数学模块：用来算平方根（即向量的 L2 长度）
import re                    # 正则模块：按规则提取英文单词、中文字
from typing import List      # 类型注解 List，用于写 List[str] / List[float]


class LightEmbedder:
    """确定性哈希词袋 embedding：把词哈希到维度空间并计数，再 L2 归一化。

    数学落点：向量（线性代数①）+ L2 范数归一化。轻量但能体现检索本质。
    纯 Python 实现，零外部依赖。
    """

    def __init__(self, dim: int = 64):
        self.dim = dim          # 向量维度（也就是"桶"的数量），默认 64

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """中英文统一分词：英文按词，中文按字 + 二元字组（无需 jieba 依赖）。

        注意：直接用 \\w+ 会把整段中文当成一个 token，导致检索失效，
        因此中文需拆成单字与相邻字组来保证重叠度。
        """
        tokens: List[str] = re.findall(r"[a-zA-Z]+", text.lower())  # 英文按单词提取并转小写
        han = re.findall(r"[一-鿿]", text)                          # 中文按单字提取（中文没有空格）
        tokens.extend(han)                                       # 先把中文单字加入词列表
        tokens.extend(han[i] + han[i + 1] for i in range(len(han) - 1))  # 再加"相邻两字"组合，增强中文重叠信号
        return tokens                                            # 返回最终的词列表

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim               # 造一个长度为 dim、全是 0.0 的列表，作为向量
        for tok in self._tokens(text):        # 遍历分出来的每一个"词"
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)  # 把词用 md5 哈希成一个大整数
            vec[h % self.dim] += 1.0          # 用"哈希值 mod 维度"定位桶，该桶计数 +1（词袋 + 哈希技巧）
        norm = math.sqrt(sum(v * v for v in vec))      # 计算向量 L2 长度（各分量平方和再开方）
        if norm > 0:                            # 防止除以 0
            vec = [v / norm for v in vec]        # 归一化：每分量都除以长度，使向量变成"单位长度"
        return vec                              # 返回归一化后的向量

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]   # 批量：对每条文本分别调用 embed


class STEmbedder:
    """sentence-transformers 真实句向量。首次使用会下载模型（属于生产模式 / 20% 留白）。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # 延迟导入：只有生产模式才需要
        except ImportError as e:  # pragma: no cover
            raise ImportError("生产模式需安装 sentence-transformers：pip install -r requirements.txt") from e
        self._model = SentenceTransformer(model_name)   # 加载真实句向量模型

    def embed(self, text: str):
        return self._model.encode(text, normalize_embeddings=True)  # 把文本编码成向量并归一化

    def embed_batch(self, texts: List[str]):
        return self._model.encode(texts, normalize_embeddings=True)  # 批量编码


def build_embedder(mode: str = "light", dim: int = 64, model_name: str = "all-MiniLM-L6-v2"):
    if mode == "light":                 # 轻量模式：教学用哈希词袋（默认）
        return LightEmbedder(dim)
    if mode == "st":                    # 生产模式：真实句向量（需先 pip 安装依赖）
        return STEmbedder(model_name)
    raise ValueError(f"未知 embedding_mode: {mode}")  # 非法模式直接报错
