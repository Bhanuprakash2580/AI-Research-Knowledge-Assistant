from fastapi import APIRouter
from ..services import analytics

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/stats")
def get_stats():
    return analytics.stats()
