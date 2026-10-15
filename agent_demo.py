"""Agent demo：演示 RAG Agent 的 Tool Use 编排。

运行：python agent_demo.py
- 若本地 RAG 服务（http://127.0.0.1:8000）在线则真实调用你的 /query /ingest /health；
- 否则自动回退到 Mock 传输，照样演示「意图 -> 选工具 -> 执行」全链路，方便面试现场展示。
"""
from __future__ import annotations

import json

from rag.agent import RagAgent, http_transport, mock_transport


def _pretty(agent: RagAgent, question: str) -> None:
    out = agent.run(question)
    print(f"\n用户：{question}")
    print(f"  -> 意图判定：{out['intent']}  |  选中工具：{out['tool_used']}")
    print(f"  -> 工具返回：{json.dumps(out['result'], ensure_ascii=False)}")


def demo_main() -> None:
    # 先探测真实服务是否在线
    probe = RagAgent(mode="demo", transport=http_transport("http://127.0.0.1:8000"))
    health = probe.run("服务在线吗")
    use_real = "error" not in health["result"]

    agent = RagAgent(mode="demo") if use_real else RagAgent(mode="demo", transport=mock_transport())
    print("=" * 54)
    print("RAG Agent (Tool Use) 演示" + ("【真实调用本地服务】" if use_real else "【Mock 模式，无需起服务】"))
    print("=" * 54)
    for q in [
        "服务还在线吗？",
        "帮我重建一下知识库索引",
        "RAG 的典型流程包含哪几步？",
        "帮我算 23*4+1",
    ]:
        _pretty(agent, q)


if __name__ == "__main__":
    demo_main()
