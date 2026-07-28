from .documents import router as documents_router
from .search import router as search_router
from .analysis import router as analysis_router
from .memory import router as memory_router
from .analytics import router as analytics_router

__all__ = ["documents_router", "search_router", "analysis_router", "memory_router", "analytics_router"]