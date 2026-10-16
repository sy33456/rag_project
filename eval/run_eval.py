"""在示例语料上运行检索效果评估（零外部依赖，轻量模式可跑）。

用法：python eval/run_eval.py
输出：每条测试题的命中情况 + 汇总指标（hit_rate@3 / precision@3 / mrr）。

评估只测「检索」环节（向量召回 + 重排），不测生成——生成质量需要人工评审，
检索质量可以用标注数据量化，这正是 metrics.py 三个指标的用途。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许从项目根目录或 eval/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import evaluate  # noqa: E402
from rag import chunker, embedder, loader, retriever, vector_store  # noqa: E402
from rag.config import Config  # noqa: E402

# ---- 测试集：问题 -> 相关片段（按「答案真实所在的 来源文件+片段序号」标注）----
# 标注方法：人工通读语料，判断答案在哪一段。注意是按内容标注，不是按检索结果凑。
TEST_SET = [
    ("Transformer 是哪一年提出的，由什么论文引入", [("transformer.txt", 0)]),
    ("自注意力里的 Q、K、V 分别代表什么", [("transformer.txt", 0)]),
    ("多头注意力为什么有用", [("transformer.txt", 0)]),
    ("Transformer 相比 RNN 有什么优势", [("transformer.txt", 1)]),
    ("向量数据库和传统数据库有什么区别", [("vector_db.txt", 0)]),
    ("什么是近似最近邻索引，HNSW 和 IVF 怎么加速", [("vector_db.txt", 0)]),
    ("常见的向量数据库有哪些，怎么选型", [("vector_db.txt", 0)]),
    ("大模型的本质是什么，训练分哪几个阶段", [("llm_basics.txt", 0)]),
    ("大模型为什么会幻觉，RAG 怎么缓解", [("llm_basics.txt", 0)]),
    ("什么是上下文窗口和 temperature", [("llm_basics.txt", 0)]),
    ("RAG 的核心思想是什么，为什么能减少幻觉", [("sample.txt", 0)]),
    ("评估 RAG 系统质量常用哪些指标", [("sample.txt", 1)]),
]


def main() -> None:
    cfg = Config()
    emb = embedder.build_embedder(cfg.embedding_mode, cfg.embedding_dim, cfg.st_model_name)
    store = vector_store.build_store(cfg.vector_store, cfg.embedding_dim)

    # 1. 建索引：切分 -> 向量化 -> 入库
    docs = loader.load_dir(cfg.data_dir)
    chunks = chunker.chunk_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
    if not chunks:
        print("语料为空，请检查 data/ 目录")
        return
    vectors = emb.embed_batch([c[0] for c in chunks])
    metas = [{"text": t, "source": s, "chunk_index": i} for t, s, i in chunks]
    store.add_batch(vectors, metas)

    # 2. 把每个片段映射成一个全局 id，方便和标注对齐
    id_of = {(m["source"], m["chunk_index"]): i for i, m in enumerate(metas)}

    print(f"语料：{len(docs)} 个文件 / {len(chunks)} 个片段"
          f"（模式={cfg.embedding_mode}，chunk_size={cfg.chunk_size}，top_k={cfg.top_k}，重排={cfg.rerank}）")
    print(f"测试集：{len(TEST_SET)} 条人工标注\n")

    k = cfg.top_k  # 指标按实际召回条数算，不夸大
    results = []
    print(f"{'问题':<24}{'top-1 即中':<10}{'首个相关位置':<8}")
    for question, relevant_pairs in TEST_SET:
        q_vec = emb.embed(question)
        hits = retriever.retrieve(store, q_vec, question, cfg.top_k, cfg.rerank)
        # 检索结果 -> 全局 id 列表；标注 -> 全局 id 集合
        ranked = [id_of[(m.get("source"), m.get("chunk_index"))] for _, m in hits]
        relevant = {id_of[p] for p in relevant_pairs}
        results.append(evaluate(ranked, relevant, k=k))
        first_pos = next((i + 1 for i, r in enumerate(ranked) if r in relevant), None)
        top1 = "√" if first_pos == 1 else "×"
        print(f"{question:<24}{top1:<10}{str(first_pos) or '-':<8}")

    n = len(results)
    hit = sum(r["hit_rate"] for r in results) / n
    prec = sum(r[f"precision@{k}"] for r in results) / n
    mrr_avg = sum(r["mrr"] for r in results) / n
    print(f"\n汇总（{n} 条测试，top_{k}）：")
    print(f"  hit_rate@{k}    = {hit:.2f}   （相关片段是否出现在前 {k} 条）")
    print(f"  precision@{k}   = {prec:.2f}   （前 {k} 条中相关片段占比）")
    print(f"  mrr           = {mrr_avg:.2f}   （首个相关片段排名的倒数均值）")
    print("\n说明：指标基于小型示例语料与人工标注，用于演示评估闭环；"
          "生产环境需更大语料与更细粒度标注。")


if __name__ == "__main__":
    main()
