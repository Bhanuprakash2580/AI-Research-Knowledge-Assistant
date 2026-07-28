from pathlib import Path
import json
from ..db import count_documents, query_stats, list_documents

def stats(index_dir: str = "index"):
    p = Path(index_dir)
    registry = p / "registry.json"
    reg = {}
    if registry.exists():
        with open(registry, "r", encoding="utf-8") as f:
            reg = json.load(f)
    total_docs = count_documents()
    processed = count_documents(status="processed")
    total_chunks = 0
    for v in reg.values():
        chunks_path = Path(v.get("chunks_file", ""))
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                total_chunks += len(json.load(f))
    docs = list_documents()
    category_distribution = {}
    for doc in docs:
        category = doc.get("category") or "unclassified"
        category_distribution[category] = category_distribution.get(category, 0) + 1
    query_metrics = query_stats()
    return {
        "total_documents": total_docs,
        "processed_documents": processed,
        "indexed_documents": len(reg),
        "total_chunks": total_chunks,
        "total_embeddings_generated": total_chunks,
        "category_distribution": category_distribution,
        **query_metrics,
    }
