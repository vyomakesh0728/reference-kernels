from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
import urllib.error
import urllib.request

from .__init__ import DEFAULT_EMBED_BASE_URL, DEFAULT_EMBED_MODEL


API_KEY_ENV_CANDIDATES = ("MIXEDBREAD_API_KEY", "MXBAI_API_KEY", "MIXEDBREAD_API_TOKEN")


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str = DEFAULT_EMBED_BASE_URL
    model: str = DEFAULT_EMBED_MODEL
    api_key: str | None = None
    prompt: str | None = None
    timeout_seconds: float = 60.0


class MixedbreadClient:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        config = config or EmbeddingConfig(api_key=_env_api_key())
        self.config = config

    @property
    def available(self) -> bool:
        return bool(self.config.api_key)

    def embed_texts(
        self,
        texts: list[str],
        *,
        is_query: bool,
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not self.config.api_key:
            raise RuntimeError(
                "Mixedbread API key not found. Set MIXEDBREAD_API_KEY or MXBAI_API_KEY to build/query dense embeddings."
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {
                "model": self.config.model,
                "input": batch,
                "normalized": True,
                "encoding_format": "float",
            }
            if self.config.prompt:
                payload["prompt"] = self.config.prompt
            response = self._post_json("/v1/embeddings", payload)
            data = response.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Mixedbread embeddings response did not contain a 'data' list")
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                    raise RuntimeError("Mixedbread embeddings response item was missing 'embedding'")
                vectors.append([float(value) for value in item["embedding"]])
        return vectors

    def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + path,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "amd-kernel-rag/1.0",
            },
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace").strip()
                last_error = RuntimeError(f"Mixedbread HTTP {exc.code}: {body or exc.reason}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= 3:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= 3:
                    raise RuntimeError(f"Mixedbread request failed: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Mixedbread request failed: {last_error}")


def env_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        base_url=os.environ.get("AMD_KERNEL_RAG_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL),
        model=os.environ.get("AMD_KERNEL_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        api_key=_env_api_key(),
        prompt=os.environ.get("AMD_KERNEL_RAG_EMBED_PROMPT") or None,
        timeout_seconds=float(os.environ.get("AMD_KERNEL_RAG_EMBED_TIMEOUT_SECONDS", "60")),
    )


def _env_api_key() -> str | None:
    for key in API_KEY_ENV_CANDIDATES:
        value = os.environ.get(key)
        if value:
            return value
    return None
