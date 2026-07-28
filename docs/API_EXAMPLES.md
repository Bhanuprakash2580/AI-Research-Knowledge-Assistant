# API Examples

Set the base URL:

```bash
set BASE_URL=http://127.0.0.1:8000
```

Upload a PDF:

```bash
curl -X POST "%BASE_URL%/documents/upload" -F "file=@sample.pdf"
```

List documents:

```bash
curl "%BASE_URL%/documents/"
```

Hybrid search:

```bash
curl -X POST "%BASE_URL%/search/" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"methodology limitations\",\"mode\":\"hybrid\",\"k\":5}"
```

Ask a grounded question:

```bash
curl -X POST "%BASE_URL%/search/qa" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What are the key findings?\",\"mode\":\"hybrid\",\"k\":5,\"session_id\":\"demo\"}"
```

Summarize a document:

```bash
curl "%BASE_URL%/analysis/summarize/DOC_ID?type=executive"
```

Compare documents:

```bash
curl -X POST "%BASE_URL%/analysis/compare?focus=methodologies" ^
  -H "Content-Type: application/json" ^
  -d "[\"DOC_ID_1\",\"DOC_ID_2\"]"
```

View analytics:

```bash
curl "%BASE_URL%/analytics/stats"
```
