from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "mllabiome-ii-api",
        "mode": "interactive_inference",
    }
