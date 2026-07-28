import os
import tempfile

TEST_DIR = tempfile.mkdtemp(prefix="ai_research_assistant_tests_")
os.environ["DATABASE_FILE"] = os.path.join(TEST_DIR, "test.db")
os.environ["INDEX_DIR"] = os.path.join(TEST_DIR, "index")
os.environ["STORAGE_DIR"] = os.path.join(TEST_DIR, "storage")
os.environ["EMBEDDING_BACKEND"] = "tfidf"

from fastapi.testclient import TestClient

from app.classifier import predict_category
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


def test_classifier_heuristic_fallback_detects_ml_and_cv_keywords():
    ml_result = predict_category("Use a dataset to train a model")
    cv_result = predict_category("A camera vision system detects objects")

    assert ml_result["category"] == "ML"
    assert cv_result["category"] == "CV"


def test_classifier_heuristic_fallback_detects_dataset_keywords():
    result = predict_category("A dataset improves prediction accuracy")

    assert result["category"] == "ML"


def test_qa_llm_error_fallback(monkeypatch):
    # Mock search_backend to return mock search results
    from app.api import search

    def mock_search(*args, **kwargs):
        return [
            {
                "doc_id": "doc1",
                "file_name": "research.pdf",
                "page_number": 1,
                "chunk_id": "chunk_0",
                "score": 0.95,
                "text": "Deep learning architectures rely on multi-layer neural networks.",
            }
        ]

    monkeypatch.setattr(search, "_get_search_backend", lambda: mock_search)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")

    # Mock OpenAI client to simulate 429 quota error
    class FakeOpenAI:
        def __init__(self, api_key):
            pass

        @property
        def chat(self):
            class Chat:
                @property
                def completions(self):
                    class Completions:
                        def create(self, **kwargs):
                            raise Exception("Error code: 429 - {'error': {'type': 'insufficient_quota'}}")

                    return Completions()

            return Chat()

    import sys
    monkeypatch.setitem(sys.modules, "openai", type("module", (), {"OpenAI": FakeOpenAI}))

    response = client.post(
        "/search/qa",
        json={"query": "What are deep learning architectures?", "mode": "semantic", "k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert "OpenAI API Quota Exceeded (Error 429)" in body["answer"]
    assert "Deep learning architectures" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["doc_id"] == "doc1"

