from pathlib import Path
import numpy as np
import json
from typing import List


class SimpleVectorStore:
    def __init__(self, index_dir: str = "index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self):
        reg = self.index_dir / "registry.json"
        if not reg.exists():
            return {}
        with open(reg, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_embeddings(self, doc_id: str):
        reg = self.list_documents()
        if doc_id not in reg:
            return None
        info = reg[doc_id]
        return np.load(info["emb_file"]), info["chunks_file"]
