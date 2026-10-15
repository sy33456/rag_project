"""API 测试：验证 /query /ingest /health 与缓存行为。

为什么用 importorskip：本项目轻量模式的设计目标是零依赖，
测试不能因为没装 fastapi 就失败（CI 里会装，本地裸跑则跳过）。
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="轻量模式零依赖，未安装 fastapi 时跳过 API 测试")

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["indexed_chunks"] > 0


def test_query_returns_sources(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "什么是余弦相似度？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"], "应返回回答"
    assert body["sources"], "应带引用来源"
    assert body["sources"][0]["score"] > 0


def test_query_cache_hit(client: TestClient) -> None:
    # 相同问题第二次应命中缓存（TTL 300s 内），结果与首次一致
    q = "RAG 的典型流程包含哪几步？"
    first = client.post("/query", json={"question": q}).json()
    second = client.post("/query", json={"question": q}).json()
    assert first == second


def test_ingest_rebuilds_index(client: TestClient) -> None:
    resp = client.post("/ingest")
    assert resp.status_code == 200
    assert resp.json()["indexed_chunks"] > 0
    # 重建后索引数应保持不变（追加语义会导致翻倍，这里防回归）
    before = client.get("/health").json()["indexed_chunks"]
    client.post("/ingest")
    after = client.get("/health").json()["indexed_chunks"]
    assert before == after
