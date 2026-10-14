"""Low-level OpenRouter transport; an unknown configured model id is a hard failure, never a silent fallback."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


class OpenRouterError(RuntimeError):
    pass


class ModelNotInCatalogError(OpenRouterError):
    """Raised when a configured model id is absent from the OpenRouter catalog."""


@dataclass
class EmbedResponse:
    vectors: list[list[float]]
    raw: dict[str, Any]
    model_returned: str


@dataclass
class ChatResponse:
    content: str
    raw: dict[str, Any]
    generation_id: str
    model_returned: str
    finish_reason: str
    # "" does NOT mean the model did not reason: some providers serve it only under `reasoning_details`.
    reasoning: str = ""
    provider: str = ""


def load_api_keys() -> list[str]:
    """Resolve the pool of OpenRouter keys for parallel suite execution."""
    multi = os.environ.get("OPENROUTER_API_KEYS", "").strip()
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if single:
        return [single]
    raise OpenRouterError("Neither OPENROUTER_API_KEYS nor OPENROUTER_API_KEY is set")


def _api_key() -> str:
    return load_api_keys()[0]


def provider_routing(providers: list[str]) -> dict[str, list[str]]:
    """OpenRouter's `provider` field: try the slugs in order, never leave the list."""
    return {"order": list(providers), "only": list(providers)}


class OpenRouterClient:
    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        api_key: str | None = None,
        providers: list[str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key
        self.providers = list(providers or [])

    def _headers(self) -> dict[str, str]:
        key = self._api_key or _api_key()
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return [m["id"] for m in data.get("data", []) if "id" in m]

    def validate_models(self, model_ids: list[str]) -> dict[str, Any]:
        """Fail unless every requested model id exists in the catalog."""
        catalog = set(self.list_models())
        missing = [m for m in dict.fromkeys(model_ids) if m not in catalog]
        if missing:
            raise ModelNotInCatalogError(
                f"Configured model id(s) not in OpenRouter catalog: {missing}"
            )
        return {
            "validated": True,
            "requested_models": list(dict.fromkeys(model_ids)),
            "catalog_size": len(catalog),
        }

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResponse:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self.providers:
            payload["provider"] = provider_routing(self.providers)
        # "none"/"off" DISABLES reasoning, so inject and traverse stay matched.
        if reasoning_effort:
            eff = reasoning_effort.lower()
            payload["reasoning"] = (
                {"enabled": False} if eff in ("none", "off") else {"effort": reasoning_effort}
            )

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError(f"No choices in response: {data}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "") or ""
        return ChatResponse(
            content=content,
            raw=data,
            generation_id=data.get("id", "") or "",
            model_returned=data.get("model", model) or model,
            finish_reason=finish_reason,
            reasoning=message.get("reasoning") or "",
            provider=data.get("provider") or "",
        )

    def generation_provider(self, generation_id: str, attempts: int = 4, timeout: float = 10.0) -> str:
        """The provider that served a finished generation, from OpenRouter's metadata."""
        # The metadata can lag the completion by a few seconds, so a miss is retried.
        url = f"{self.base_url}/generation"
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.get(url, headers=self._headers(), params={"id": generation_id})
                    resp.raise_for_status()
                    name = ((resp.json() or {}).get("data") or {}).get("provider_name") or ""
                if name:
                    return name
                err: Exception = OpenRouterError(f"no provider_name for {generation_id}")
            except (httpx.HTTPError, ValueError) as e:
                err = e
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
        raise err

    def embed(self, model: str, texts: list[str]) -> EmbedResponse:
        """Embed a batch of texts, returning one vector per input in order."""
        url = f"{self.base_url}/embeddings"
        # No provider pin: the chat-model list would make OpenRouter answer 404 for an embedding model.
        payload: dict[str, Any] = {
            "model": model,
            "input": list(texts),
            "usage": {"include": True},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("data") or []
        if len(items) != len(texts):
            raise OpenRouterError(f"Expected {len(texts)} embeddings, got {len(items)}: {data}")
        vectors = [list(item["embedding"]) for item in sorted(items, key=lambda i: i.get("index", 0))]
        return EmbedResponse(
            vectors=vectors, raw=data, model_returned=data.get("model", model) or model,
        )
