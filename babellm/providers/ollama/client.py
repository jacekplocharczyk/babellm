"""Ollama provider implementation."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..._http import HTTPClient
from ...base.provider import BaseLLMProvider
from ...types import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    GenerateChunk,
    GenerateResponse,
    Message,
)
from . import _serializers as ser

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseLLMProvider):
    """Ollama REST API provider."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._http = HTTPClient(base_url, headers=headers, timeout=timeout)

    def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint."""
        payload = self._build_chat_payload(
            messages,
            model,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs,
        )
        raw = self._http.post("/api/chat", payload)
        return ser.ollama_to_chat_response(raw, model)

    def chat_stream(
        self,
        messages: list[Message],
        model: str,
        **kwargs: Any,
    ) -> Iterator[ChatChunk]:
        """Stream chat endpoint."""
        payload = self._build_chat_payload(messages, model, stream=True, **kwargs)
        for raw_chunk in self._http.post_stream("/api/chat", payload):
            yield ser.ollama_to_chat_chunk(raw_chunk, model)

    def generate(
        self,
        prompt: str,
        model: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> GenerateResponse:
        """Generate endpoint."""
        payload = self._build_generate_payload(
            prompt,
            model,
            stream=False,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        raw = self._http.post("/api/generate", payload)
        return ser.ollama_to_generate_response(raw, model)

    def generate_stream(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> Iterator[GenerateChunk]:
        """Stream generate endpoint."""
        payload = self._build_generate_payload(prompt, model, stream=True, **kwargs)
        for raw_chunk in self._http.post_stream("/api/generate", payload):
            yield ser.ollama_to_generate_chunk(raw_chunk, model)

    def embed(
        self,
        input: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Embed endpoint."""
        input_list = input if isinstance(input, list) else [input]
        payload = {"model": model, "input": input_list}
        payload.update(kwargs)
        raw = self._http.post("/api/embed", payload)
        return ser.ollama_to_embedding_response(raw, model)

    async def achat(
        self,
        messages: list[Message],
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat endpoint."""
        payload = self._build_chat_payload(messages, model, stream=False, **kwargs)
        raw = await self._http.apost("/api/chat", payload)
        return ser.ollama_to_chat_response(raw, model)

    async def achat_stream(
        self,
        messages: list[Message],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        """Async stream chat endpoint."""
        payload = self._build_chat_payload(messages, model, stream=True, **kwargs)
        async for raw_chunk in self._http.apost_stream("/api/chat", payload):
            yield ser.ollama_to_chat_chunk(raw_chunk, model)

    async def agenerate(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> GenerateResponse:
        """Async generate endpoint."""
        payload = self._build_generate_payload(prompt, model, stream=False, **kwargs)
        raw = await self._http.apost("/api/generate", payload)
        return ser.ollama_to_generate_response(raw, model)

    async def agenerate_stream(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[GenerateChunk]:
        """Async stream generate endpoint."""
        payload = self._build_generate_payload(prompt, model, stream=True, **kwargs)
        async for raw_chunk in self._http.apost_stream("/api/generate", payload):
            yield ser.ollama_to_generate_chunk(raw_chunk, model)

    async def aembed(
        self,
        input: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Async embed endpoint."""
        input_list = input if isinstance(input, list) else [input]
        payload = {"model": model, "input": input_list}
        payload.update(kwargs)
        raw = await self._http.apost("/api/embed", payload)
        return ser.ollama_to_embedding_response(raw, model)

    def close(self) -> None:
        """Close HTTP client."""
        self._http.close()

    async def aclose(self) -> None:
        """Close async HTTP client."""
        await self._http.aclose()

    def __enter__(self) -> OllamaProvider:
        """Context manager entry."""
        self._http.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self._http.__exit__(*args)

    async def __aenter__(self) -> OllamaProvider:
        """Async context manager entry."""
        await self._http.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self._http.__aexit__(*args)

    @staticmethod
    def _build_chat_payload(
        messages: list[Message],
        model: str,
        stream: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build chat payload for Ollama API."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": ser.messages_to_ollama(messages),
            "stream": stream,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["num_predict"] = max_tokens
        if stop is not None:
            payload["stop"] = stop

        payload.update(kwargs)
        return payload

    @staticmethod
    def _build_generate_payload(
        prompt: str,
        model: str,
        stream: bool,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build generate payload for Ollama API."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        if system is not None:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["num_predict"] = max_tokens

        payload.update(kwargs)
        return payload
