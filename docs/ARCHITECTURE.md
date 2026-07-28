# Architecture

The backend follows a simple modular FastAPI structure:

```text
app/
  api/          REST route modules
  services/     document processing, search, summaries, analytics, memory
  ml/           classifier training script
  db.py         SQLite schema and data access helpers
  classifier.py runtime category prediction
  main.py       FastAPI application bootstrap
```

## Processing Flow

1. The user uploads a PDF through the documents API.
2. Metadata is inserted into SQLite with `uploaded` status.
3. The processor extracts page text from the PDF.
4. Text is cleaned and split into overlapping page-aware chunks.
5. Chunks are embedded using sentence-transformers when available, or TF-IDF fallback.
6. Embeddings and chunk metadata are written into the local index directory.
7. The classifier predicts a technical category.
8. SQLite metadata is updated with page count, chunk count, status, and category.

## Query Flow

1. The user sends a search or QA request.
2. The retrieval layer loads indexed chunks.
3. Results are ranked using keyword, semantic, or hybrid scoring.
4. QA builds a context block with document, page, chunk, and score metadata.
5. If OpenAI is configured, the answer is generated from retrieved context only.
6. If OpenAI is unavailable, the endpoint returns the most relevant grounded excerpts.

## Persistence

- SQLite stores documents, conversations, and query analytics.
- The local index stores chunk JSON and embedding `.npy` files.
- Uploaded PDFs are stored in `STORAGE_DIR`.

## Scalability Path

For a production deployment, the file-backed vector index can be replaced with FAISS, Chroma, Qdrant, Pinecone, or another vector database without changing the public API contract.
