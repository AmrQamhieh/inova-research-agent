import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    # `app.database.get_database_url()` raises if no DB env is set, and `app.main`
    # touches the DB at import time. Force a harmless DB URL for import.
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")

    import app.database as database

    database.get_database_url.cache_clear()
    database.get_engine.cache_clear()

    import app.main as main

    importlib.reload(main)
    return TestClient(main.app)


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Inova Research Agent API is running."}


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK!"}


def test_query_returns_queryresponse_schema(client: TestClient, mocker):
    # Avoid DB access inside `build_thread_messages` and when persisting conversation.
    mocker.patch("app.main.build_thread_messages", return_value=[])

    class DummySession:
        def add(self, _obj):  # noqa: ANN001
            return None

        def flush(self):
            return None

        def commit(self):
            return None

        def refresh(self, _obj):  # noqa: ANN001
            return None

        def close(self):
            return None

    mocker.patch("app.main.SessionLocal", return_value=DummySession())

    mocker.patch(
        "app.main.agent_graph.invoke",
        return_value={"messages": [SimpleNamespace(content="Mocked assistant response")]} ,
    )

    payload = {"thread_id": "test-thread", "text": "Hello"}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Response should match `QueryResponse` schema: { thread_id: str, response: str }
    assert data == {"thread_id": "test-thread", "response": "Mocked assistant response"}


def test_query_empty_text_returns_400(client: TestClient):
    payload = {"thread_id": "test-thread", "text": "   "}
    response = client.post("/query", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Input text cannot be empty."


def test_query_missing_text_returns_422(client: TestClient):
    payload = {"thread_id": "test-thread"}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


@patch("app.main.SessionLocal")
def test_history_endpoint_returns_shape(_mock_sessionlocal, client: TestClient, mocker):
    class DummyQuery:
        def filter(self, *_args, **_kwargs):
            return self

        def order_by(self, *_args, **_kwargs):
            return self

        def all(self):
            return []

    class DummySession:
        def query(self, *_args, **_kwargs):
            return DummyQuery()

        def close(self):
            return None

    mocker.patch("app.main.SessionLocal", return_value=DummySession())

    response = client.get("/history/test-thread")
    assert response.status_code == 200

    data = response.json()
    assert data["thread_id"] == "test-thread"
    assert isinstance(data["messages"], list)