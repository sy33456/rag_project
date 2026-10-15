"""全局配置：所有可调参数集中在此，改这里即可切换模式。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # 切分
    chunk_size: int = 500        # 每个片段最大字符数
    chunk_overlap: int = 50      # 片段间重叠，避免截断语义

    # 检索
    top_k: int = 3               # 召回片段数
    rerank: bool = True          # 是否做关键词重排

    # Embedding
    embedding_mode: str = "light"   # "light" 或 "st"（sentence-transformers）
    embedding_dim: int = 64         # 轻量模式维度
    st_model_name: str = "all-MiniLM-L6-v2"

    # 向量库
    vector_store: str = "numpy"     # "numpy" 或 "chromadb"

    # LLM（生产模式）
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # 数据目录
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量 / .env 读取 LLM 配置。"""
        cfg = cls()
        cfg.llm_api_key = os.getenv("LLM_API_KEY", "")
        cfg.llm_base_url = os.getenv("LLM_BASE_URL", cfg.llm_base_url)
        cfg.llm_model = os.getenv("LLM_MODEL", cfg.llm_model)
        return cfg

    def use_prod(self) -> "Config":
        """切换到生产模式（真实 embedding + 向量库）。"""
        self.embedding_mode = "st"
        self.vector_store = "chromadb"
        return self
