"""CLI 演示：建索引 -> 问答。轻量模式零依赖即可运行。

用法：
    python main.py                # 轻量模式
    python main.py --mode prod    # 生产模式（需 pip install -r requirements.txt 并配置 .env）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import Config
from rag.loader import load_dir
from rag.pipeline import RAGPipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 文档问答骨架演示")
    ap.add_argument("--mode", choices=["light", "prod"], default="light")
    args = ap.parse_args()

    cfg = Config.from_env()
    if args.mode == "prod":
        cfg.use_prod()

    data_dir = cfg.data_dir
    docs = load_dir(data_dir)
    if not docs:
        print(f"未在 {data_dir} 找到 .txt/.md 语料，请先放入文档。")
        return 1

    pipe = RAGPipeline(cfg)
    n = pipe.index(docs)
    print(f"[索引完成] 来源 {len(docs)} 篇，切分 {n} 个片段，模式={cfg.embedding_mode}/{cfg.vector_store}")

    print("\n输入问题（空行或 Ctrl-C 退出）：")
    try:
        while True:
            q = input(">> ").strip()
            if not q:
                break
            res = pipe.ask(q)
            print("\n--- 回答 ---")
            print(res["answer"])
            print("\n--- 引用来源 ---")
            for s in res["sources"]:
                print(f"  {s['source']}#{s['chunk_index']}  相似度={s['score']}")
            print()
    except (EOFError, KeyboardInterrupt):
        print("\n再见。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
