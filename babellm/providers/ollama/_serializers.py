"""Serialization functions for Ollama API responses."""

from typing import Any

from ...types import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    GenerateChunk,
    GenerateResponse,
    Message,
    Role,
    Usage,
)


def messages_to_ollama(messages: list[Message]) -> list[dict[str, str]]:
    """Convert babellm Messages to Ollama API format."""
    return [{"role": msg.role.value, "content": msg.content} for msg in messages]


def ollama_to_usage(raw: dict[str, Any]) -> Usage:
    """Extract usage info from Ollama response."""
    prompt_tokens = raw.get("prompt_eval_count", 0)
    completion_tokens = raw.get("eval_count", 0)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def ollama_to_chat_response(raw: dict[str, Any], model: str) -> ChatResponse:
    """Convert Ollama chat response to ChatResponse."""
    message_data = raw.get("message", {})
    message = Message(
        role=Role(message_data.get("role", "assistant")),
        content=message_data.get("content", ""),
    )
    return ChatResponse(
        message=message,
        model=model,
        done=raw.get("done", False),
        usage=ollama_to_usage(raw),
        raw=raw,
    )


def ollama_to_chat_chunk(raw: dict[str, Any], model: str) -> ChatChunk:
    """Convert Ollama streaming chat response to ChatChunk."""
    message_data = raw.get("message", {})
    content = message_data.get("content", "")
    return ChatChunk(
        delta=content,
        model=model,
        done=raw.get("done", False),
    )


def ollama_to_generate_response(raw: dict[str, Any], model: str) -> GenerateResponse:
    """Convert Ollama generate response to GenerateResponse."""
    return GenerateResponse(
        text=raw.get("response", ""),
        model=model,
        done=raw.get("done", False),
        usage=ollama_to_usage(raw),
        raw=raw,
    )


def ollama_to_generate_chunk(raw: dict[str, Any], model: str) -> GenerateChunk:
    """Convert Ollama streaming generate response to GenerateChunk."""
    return GenerateChunk(
        delta=raw.get("response", ""),
        model=model,
        done=raw.get("done", False),
    )


def ollama_to_embedding_response(raw: dict[str, Any], model: str) -> EmbeddingResponse:
    """Convert Ollama embedding response to EmbeddingResponse."""
    embedding = raw.get("embedding", raw.get("embeddings", [{}])[0] if raw.get("embeddings") else [])
    return EmbeddingResponse(
        embedding=embedding,
        model=model,
        raw=raw,
    )
