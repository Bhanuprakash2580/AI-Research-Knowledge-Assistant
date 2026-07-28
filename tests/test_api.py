import os
import tempfile

TEST_DIR = tempfile.mkdtemp(prefix="ai_research_assistant_tests_")
os.environ["DATABASE_FILE"] = os.path.join(TEST_DIR, "test.db")
os.environ["INDEX_DIR"] = os.path.join(TEST_DIR, "index")
os.environ["STORAGE_DIR"] = os.path.join(TEST_DIR, "storage")
os.environ["EMBEDDING_BACKEND"] = "tfidf"

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analytics_empty_state():
    response = client.get("/analytics/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_documents"] == 0
    assert body["total_chunks"] == 0


def test_qa_without_documents_is_grounded_fallback():
    response = client.post(
        "/search/qa",
        json={"query": "What does the document say?", "mode": "semantic", "k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "I cannot determine the answer from the provided documents."
    assert body["sources"] == []
