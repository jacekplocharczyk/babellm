"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from ..types import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    GenerateChunk,
    GenerateResponse,
    Message,
)


class BaseLLMProvider(ABC):
    """
    Abstract base class that defines the contract for all LLM providers.

    Every provider must implement both sync and async variants of:
    - chat: multi-turn conversation
    - chat_stream: streaming chat
    - generate: single-shot text generation
    - generate_stream: streaming generation
    - embed: embeddings
    """

    # --- Sync API ---

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **kwargs: dict,
    ) -> ChatResponse:
        """
        Chat endpoint.

        Args:
            messages: List of messages in conversation.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            stop: Stop sequences.
            **kwargs: Provider-specific options.
        """

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        model: str,
        **kwargs: dict,
    ) -> Iterator[ChatChunk]:
        """Stream chat response."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: dict,
    ) -> GenerateResponse:
        """Single-shot text generation."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        model: str,
        **kwargs: dict,
    ) -> Iterator[GenerateChunk]:
        """Stream generation response."""

    @abstractmethod
    def embed(
        self,
        input: str | list[str],
        model: str,
        **kwargs: dict,
    ) -> EmbeddingResponse:
        """Generate embeddings."""

    # --- Async API ---

    @abstractmethod
    async def achat(
        self,
        messages: list[Message],
        model: str,
        **kwargs: dict,
    ) -> ChatResponse:
        """Async chat endpoint."""

    @abstractmethod
    async def achat_stream(
        self,
        messages: list[Message],
        model: str,
        **kwargs: dict,
    ) -> AsyncIterator[ChatChunk]:
        """Async stream chat response."""

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        model: str,
        **kwargs: dict,
    ) -> GenerateResponse:
        """Async single-shot text generation."""

    @abstractmethod
    async def agenerate_stream(
        self,
        prompt: str,
        model: str,
        **kwargs: dict,
    ) -> AsyncIterator[GenerateChunk]:
        """Async stream generation response."""

    @abstractmethod
    async def aembed(
        self,
        input: str | list[str],
        model: str,
        **kwargs: dict,
    ) -> EmbeddingResponse:
        """Async generate embeddings."""
