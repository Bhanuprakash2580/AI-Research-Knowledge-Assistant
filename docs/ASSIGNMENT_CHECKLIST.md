# Assignment Checklist

| Requirement | Project Support |
| --- | --- |
| Upload PDF documents | `POST /documents/upload` and `POST /documents/upload-batch` |
| Store document metadata | SQLite `documents` table |
| List and delete documents | `GET /documents/`, `GET /documents/{doc_id}`, `DELETE /documents/{doc_id}` |
| Reprocess documents | `POST /documents/{doc_id}/reprocess` |
| Text extraction | `app/services/processor.py` using PyPDF2 |
| Intelligent chunking | Page-aware overlapping chunks |
| Embedding generation | sentence-transformers when available, TF-IDF fallback |
| Vector indexing | Local file-backed index in `INDEX_DIR` |
| Keyword search | `mode=keyword` |
| Semantic search | `mode=semantic` |
| Hybrid search | `mode=hybrid` |
| RAG QA with citations | `POST /search/qa` |
| Summarization | `GET /analysis/summarize/{doc_id}` |
| Document comparison | `POST /analysis/compare` |
| TensorFlow classification | `app/ml/train_classifier.py` and `app/classifier.py` |
| Conversation memory | `/memory/*` endpoints and QA `session_id` |
| Analytics | `GET /analytics/stats` |
| API documentation | FastAPI Swagger and Postman collection |
| Docker support | `Dockerfile` |
| Tests | `tests/test_api.py` |

## Notes

The backend is designed to remain runnable without paid API keys. When `OPENAI_API_KEY` is not set, QA, summaries, and comparisons return grounded extractive fallbacks instead of hallucinated answers.
