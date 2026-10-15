"""Agent 模块：基于 RAG 服务的 Tool Use / Function Calling 编排。

设计取舍（面试可讲）：
1. 为什么在 RAG 之上加 Agent：纯 RAG 是「检索 -> 生成」的单次流水线，用户只能问问题；
   Agent 让系统「自己决定调哪个工具、调几次」，能处理多步任务
   （例如：先查健康检查确认服务在线，再检索，再根据结果决定是否补充入库）。
2. 为什么用 Tool Use / Function Calling 而不是硬编码 if-else 流程：
   硬编码只能走固定路径；把能力封装成「工具」交给 LLM 按需编排，
   新增一个工具（如联网搜索）即可扩展，无需改主流程。
3. 为什么保留 demo 意图路由模式：真实 LLM 需要 key 且偶有不稳定；
   demo 用关键词路由把「意图识别 -> 工具选择 -> 执行 -> 汇总」的架构跑通，
   既方便面试现场演示，也便于做无网络的单测。
4. 工具层为什么用内置 urllib 而非 requests：保持轻量零依赖，
   与项目「裸跑也能演示」的定位一致（requests 在 generator 里才是延迟导入）。
"""
from __future__ import annotations

import ast
import json
import operator
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


# ---------- 传输层：把「调用工具」统一成一次 HTTP 请求 ----------

def http_transport(base_url: str) -> Callable[[str, str, dict | None], dict]:
    """返回一个 transport：给定 method / path / body，向 RAG 服务发请求并返回 JSON。

    为什么抽一层 transport：真实调用走 HTTP（调用你已有的 FastAPI 接口），
    但测试 / 演示可以替换成 Mock，无需真的起服务。
    """
    base = base_url.rstrip("/")

    def _send(method: str, path: str, body: dict | None) -> dict:
        url = base + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:  # 连接失败等
            return {"error": str(e)}

    return _send


def mock_transport(health_chunks: int = 12) -> Callable[[str, str, dict | None], dict]:
    """用于演示 / 测试的假传输：模拟 RAG 服务返回，不真正发网络请求。"""
    def _send(method: str, path: str, body: dict | None) -> dict:
        if path == "/health":
            return {"status": "ok", "indexed_chunks": health_chunks}
        if path == "/ingest":
            return {"indexed_chunks": health_chunks}
        if path == "/query":
            return {
                "question": (body or {}).get("question", ""),
                "answer": f"【演示】根据知识库，关于「{(body or {}).get('question', '')}」的回答……",
                "sources": [{"score": 0.87, "source": "demo.md", "chunk_index": 0}],
            }
        return {"error": "unknown path"}
    return _send


# ---------- 工具定义 ----------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    fn: Callable[[dict], dict]  # 接收参数 dict，返回结果 dict


def make_tools(transport: Callable[[str, str, dict | None], dict]) -> list[Tool]:
    """把 RAG 服务的三个接口 + 一个外部工具封装成 Agent 可调用工具。"""
    def rag_query(args: dict) -> dict:
        return transport("POST", "/query", {"question": args["question"]})

    def rag_ingest(args: dict) -> dict:
        return transport("POST", "/ingest", None)

    def rag_health(args: dict) -> dict:
        return transport("GET", "/health", None)

    def calculator(args: dict) -> dict:
        # 演示「Agent 不止会查 RAG，还能调任意工具」：仅允许四则运算，规避 eval 风险
        # 输入可能是中文句子（如"帮我算 23*4+1"），先抽出数学部分再算
        return {"result": _safe_eval(_extract_expr(args["expression"]))}

    return [
        Tool(
            name="rag_query",
            description="向知识库提问，返回带引用来源的回答。当用户想了解资料内容时使用。",
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string", "description": "用户的问题"}},
                "required": ["question"],
            },
            fn=rag_query,
        ),
        Tool(
            name="rag_ingest",
            description="重建知识库索引（语料更新后调用）。当用户要求更新/重建/导入知识库时使用。",
            parameters={"type": "object", "properties": {}},
            fn=rag_ingest,
        ),
        Tool(
            name="rag_health",
            description="检查 RAG 服务是否在线及已索引片段数。当用户问服务状态/健康时使用。",
            parameters={"type": "object", "properties": {}},
            fn=rag_health,
        ),
        Tool(
            name="calculator",
            description="计算一个四则运算表达式。当用户需要做算术时使用（演示非 RAG 工具）。",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "如 '23*4+1'"}},
                "required": ["expression"],
            },
            fn=calculator,
        ),
    ]


def _looks_like_math(text: str) -> bool:
    """粗略判断是否为算式：含数字且含四则运算符（demo 路由用）。"""
    has_digit = any(c.isdigit() for c in text)
    has_op = any(op in text for op in ["+", "-", "*", "/", "×", "÷"])
    return has_digit and has_op


def _extract_expr(text: str) -> str:
    """从可能夹带中文的句子里抽出纯数学表达式（如"帮我算 23*4+1" -> "23*4+1"）。"""
    cleaned = "".join(re.findall(r"[0-9+\-*/().\s×÷]", text))
    return cleaned.replace("×", "*").replace("÷", "/").strip()


def _safe_eval(expr: str) -> float | str:
    """只允许数字与 + - * / ( ) 的四则运算，杜绝 eval 的安全风险。"""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持四则运算")

    try:
        return _eval(ast.parse(expr, mode="eval"))
    except Exception as e:  # noqa: BLE001 - 演示用，统一返回失败信息
        return f"计算失败: {e}"


# ---------- Agent ----------

class RagAgent:
    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 mode: str = "demo",
                 llm_base_url: str = "", llm_api_key: str = "", llm_model: str = "gpt-4o-mini",
                 max_iter: int = 5, transport=None):
        self.transport = transport or http_transport(base_url)
        self.tools = make_tools(self.transport)
        self.tool_map = {t.name: t for t in self.tools}
        self.mode = mode
        self.llm_base_url = (llm_base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "")
        self.llm_model = llm_model
        self.max_iter = max_iter

    def run(self, user_input: str) -> dict:
        # 没有 key 时即便声明 llm 模式也回退到 demo，保证不崩
        if self.mode == "llm" and self.llm_api_key:
            return self._run_llm(user_input)
        return self._run_demo(user_input)

    # ---- demo 模式：关键词意图路由 ----
    def _run_demo(self, user_input: str) -> dict:
        text = user_input.lower()
        if any(k in text for k in ["健康", "health", "在线", "状态", "活着"]):
            tool = self.tool_map["rag_health"]
            intent = "health"
        elif any(k in text for k in ["入库", "导入", "重建", "更新知识库", "更新索引", "ingest", "重新索引"]):
            tool = self.tool_map["rag_ingest"]
            intent = "ingest"
        elif any(k in text for k in ["算", "calculate", "计算器"]) or _looks_like_math(text):
            tool = self.tool_map["calculator"]
            intent = "calculator"
        else:
            tool = self.tool_map["rag_query"]
            intent = "query"
        # 不同工具需要的参数不同：query 要 question，calculator 要 expression，其余无参
        if intent == "query":
            args = {"question": user_input}
        elif intent == "calculator":
            args = {"expression": user_input}
        else:
            args = {}
        result = tool.fn(args)
        return {"mode": "demo", "intent": intent, "tool_used": tool.name, "result": result}

    # ---- LLM 模式：标准 Function Calling 循环 ----
    def _run_llm(self, user_input: str) -> dict:
        system = ("你是一个 RAG 知识库助手，可以调用工具来回答问题。"
                  "优先用 rag_query 查资料；用户要求更新知识库时用 rag_ingest；"
                  "问服务状态时用 rag_health；需要算数时用 calculator。")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        tools_schema = [
            {"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self.tools
        ]
        trace: list[dict] = []
        for _ in range(self.max_iter):
            resp = self._chat(messages, tools_schema)
            msg = resp["choices"][0]["message"]
            if not msg.get("tool_calls"):
                return {"mode": "llm", "answer": msg.get("content", ""), "tool_trace": trace}
            messages.append(msg)  # assistant 消息要带着 tool_calls 回传
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool = self.tool_map.get(fn["name"])
                out = tool.fn(args) if tool else {"error": "unknown tool"}
                trace.append({"tool": fn["name"], "args": args, "result": out})
                messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "content": json.dumps(out, ensure_ascii=False),
                })
        return {"mode": "llm", "answer": "（达到最大轮次，未完成）", "tool_trace": trace}

    def _chat(self, messages: list, tools_schema: list) -> dict:
        payload = {
            "model": self.llm_model, "messages": messages,
            "tools": tools_schema, "tool_choice": "auto",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.llm_base_url}/chat/completions", data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    # 直接运行可看到 demo 路由效果（用 Mock 传输，无需起服务）
    agent = RagAgent(mode="demo", transport=mock_transport())
    for q in ["服务还在线吗？", "帮我重建一下知识库", "RAG 的典型流程是什么？", "帮我算 23*4+1"]:
        print(agent.run(q))
