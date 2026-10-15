"""Agent 测试：验证工具定义、demo 意图路由、calculator 安全、LLM 函数调用循环。

为什么用 mock_transport：Agent 的工具本质是发 HTTP 请求给 RAG 服务，
测试不依赖起服务，用 Mock 传输把「网络」替换为可控的假返回即可单测编排逻辑。
"""
from __future__ import annotations

import pytest

from rag.agent import RagAgent, _safe_eval, make_tools, mock_transport


def test_tools_schema_well_formed() -> None:
    tools = make_tools(mock_transport())
    names = {t.name for t in tools}
    assert {"rag_query", "rag_ingest", "rag_health", "calculator"} <= names
    for t in tools:  # 每个工具都应有 JSON Schema 描述，才能交给 LLM
        assert "type" in t.parameters


def test_demo_routes_health() -> None:
    a = RagAgent(mode="demo", transport=mock_transport())
    out = a.run("服务在线吗")
    assert out["intent"] == "health"
    assert out["tool_used"] == "rag_health"
    assert out["result"]["status"] == "ok"


def test_demo_routes_ingest() -> None:
    a = RagAgent(mode="demo", transport=mock_transport())
    out = a.run("帮我重建知识库")
    assert out["intent"] == "ingest"
    assert out["tool_used"] == "rag_ingest"
    assert out["result"]["indexed_chunks"] > 0


def test_demo_routes_query_by_default() -> None:
    a = RagAgent(mode="demo", transport=mock_transport())
    out = a.run("什么是 RAG？")
    assert out["intent"] == "query"
    assert out["tool_used"] == "rag_query"
    assert out["result"]["sources"]  # 应带引用来源


def test_calculator_tool() -> None:
    a = RagAgent(mode="demo", transport=mock_transport())
    out = a.run("帮我算 23*4+1")
    assert out["tool_used"] == "calculator"
    assert out["result"]["result"] == 93


def test_safe_eval_rejects_arbitrary_code() -> None:
    # 必须拒绝 __import__ 等危险表达式，证明没用裸 eval
    bad = _safe_eval("__import__('os').system('echo hi')")
    assert isinstance(bad, str) and "计算失败" in bad


def test_llm_mode_tool_loop(monkeypatch) -> None:
    # 用假 LLM 验证「Tool Use 循环」：第一轮要求调 rag_query，第二轮给最终回答
    a = RagAgent(mode="llm", llm_api_key="fake", transport=mock_transport())
    responses = iter([
        {"choices": [{"message": {
            "content": None,
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "rag_query", "arguments": '{"question":"什么是RAG"}'},
            }],
        }}]},
        {"choices": [{"message": {"content": "根据资料，RAG 是检索增强生成……", "tool_calls": None}}]},
    ])
    monkeypatch.setattr(a, "_chat", lambda messages, tools_schema: next(responses))
    out = a.run("什么是 RAG？")
    assert out["mode"] == "llm"
    assert out["answer"]
    assert out["tool_trace"][0]["tool"] == "rag_query"
