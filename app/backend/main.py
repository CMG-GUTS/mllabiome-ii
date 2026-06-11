from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api import chat, health, inference
from app.backend.config import get_settings

settings = get_settings()

app = FastAPI(
    title="mllabiome-ii Interactive Inference API",
    description="Inference-only API for packaged MPMA-E microbiome deployment models.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(inference.router, prefix="/api", tags=["inference"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    return {
        "message": "mllabiome-ii Interactive Inference API",
        "version": "0.1.0",
        "mode": "interactive_inference",
    }
