from fastapi import FastAPI
from .api import documents, search, analysis, memory, analytics
from .db import initialize_database

app = FastAPI(title="AI Research & Knowledge Assistant")

initialize_database()

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(analysis.router)
app.include_router(memory.router)
app.include_router(analytics.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Research & Knowledge Assistant backend"}
