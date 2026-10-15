"""生成：把检索到的上下文 + 问题拼成 Prompt，调用 LLM。

- 有 LLM_API_KEY：走 OpenAI 兼容接口（GPT / Ollama / 通义等）。
- 无 key（轻量演示）：用模板兜底，把上下文直接拼回，便于看流水线。
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple


PROMPT_TEMPLATE = """你是一个严谨的问答助手。只根据下面的【上下文】回答，若上下文不足以回答请说"资料中未提及"。
【上下文】
{context}

【问题】
{question}

【回答】"""


class Generator:
    def __init__(self, base_url: str = "", api_key: str = "", model: str = "gpt-4o-mini"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        import requests  # 延迟导入，轻量模式不依赖
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def generate(self, question: str, contexts: List[Tuple[float, dict]]) -> str:
        ctx_text = "\n\n".join(
            f"[来源 {m.get('source')}#{m.get('chunk_index')}] {m.get('text')}" for _, m in contexts
        )
        prompt = PROMPT_TEMPLATE.format(context=ctx_text, question=question)

        if self.api_key:
            try:
                return self._call_llm(prompt)
            except Exception as e:  # 失败兜底，保证 demo 不崩
                return f"（LLM 调用失败：{e}）\n\n以下是检索到的上下文：\n{ctx_text}"

        # 轻量演示：无 key 时直接返回拼接上下文，直观展示 RAG 已检索到什么
        return f"（演示模式：未配置 LLM_API_KEY，以下为检索到的上下文）\n\n{ctx_text}"
