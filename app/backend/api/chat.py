from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.backend.config import get_settings

router = APIRouter()

SYSTEM_PROMPT = """You are a microbiome analysis assistant. You help interpret gut microbiome analysis results.

Rules:
- Answer based on the analysis data provided in the user message
- Explain the biological significance of bacterial genera mentioned
- Be concise (under 200 words)
- Never make diagnostic claims
- Do not ask for more data if it's already provided"""


class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    history: Optional[list[dict[str, Any]]] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model: str


class ChatConfigResponse(BaseModel):
    provider: str = "ollama"
    ollama_url: str
    default_model: str
    available_models: list[str]
    status: str
    detail: Optional[str] = None


def _safe_history(history: Optional[list[dict[str, Any]]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if not history:
        return messages
    for msg in history:
        role = str(msg.get("role", "user"))
        if role not in {"user", "assistant"}:
            role = "user"
        content = str(msg.get("content", ""))
        if content.strip():
            messages.append({"role": role, "content": content})
    return messages


def _response_detail(response: httpx.Response, model: str) -> str:
    body = response.text.strip()
    if len(body) > 500:
        body = body[:500] + "..."
    if body:
        return (
            f"Ollama returned HTTP {response.status_code} for model '{model}': {body}"
        )
    return f"Ollama returned HTTP {response.status_code} for model '{model}'."


@router.get("/chat/config", response_model=ChatConfigResponse)
async def chat_config():
    """Return Ollama assistant defaults and any locally/cloud-available model names."""
    settings = get_settings()
    available_models: list[str] = []
    status = "unavailable"
    detail: Optional[str] = None

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.ollama_url}/api/tags")
        if response.status_code == 200:
            payload = response.json()
            for item in payload.get("models", []):
                name = item.get("name") or item.get("model")
                if name:
                    available_models.append(str(name))
            available_models = sorted(set(available_models))
            status = "ok"
        else:
            detail = _response_detail(response, settings.ollama_model)
    except httpx.ConnectError:
        detail = "Cannot connect to Ollama. Start Ollama and verify OLLAMA_URL."
    except httpx.TimeoutException:
        detail = "Timed out while checking Ollama."
    except Exception as exc:  # noqa: BLE001 - endpoint should report status, not fail page load
        detail = str(exc)

    if settings.ollama_model not in available_models:
        # Cloud or not-yet-pulled models may not appear in /api/tags. Keep the
        # configured default selectable so users can still use Ollama cloud names.
        available_models.insert(0, settings.ollama_model)

    return ChatConfigResponse(
        ollama_url=settings.ollama_url,
        default_model=settings.ollama_model,
        available_models=available_models,
        status=status,
        detail=detail,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    settings = get_settings()
    model = (request.model or settings.ollama_model).strip() or settings.ollama_model

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_safe_history(request.history))

    user_content = request.message.strip()
    if request.context and request.context.strip():
        user_content = f"Analysis data:\n{request.context}\n\nQuestion: {user_content}"

    if not user_content:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    messages.append({"role": "user", "content": user_content})

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            response = await client.post(
                f"{settings.ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": settings.ollama_temperature,
                        "top_p": settings.ollama_top_p,
                        "num_predict": settings.ollama_num_predict,
                    },
                },
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502, detail=_response_detail(response, model)
            )

        data = response.json()
        assistant_message = data.get("message", {}).get("content", "")
        if not assistant_message:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama returned an empty response for model '{model}'.",
            )

        return ChatResponse(response=assistant_message, model=model)

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, detail=f"Ollama timed out while using model '{model}'."
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot connect to Ollama at {settings.ollama_url}. "
                "Start Ollama, or set OLLAMA_URL to the correct host."
            ),
        )
    except Exception as exc:  # noqa: BLE001 - bubble useful backend detail to frontend
        raise HTTPException(status_code=500, detail=str(exc))
