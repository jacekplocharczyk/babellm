"""Shared data types used across all providers."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: str


@dataclass
class Usage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    """Response from a chat endpoint."""

    message: Message
    model: str
    done: bool
    usage: Usage | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResponse:
    """Response from a text generation endpoint."""

    text: str
    model: str
    done: bool
    usage: Usage | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Response from an embedding endpoint."""

    embedding: list[float]
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatChunk:
    """Streaming chunk from a chat endpoint."""

    delta: str
    model: str
    done: bool


@dataclass
class GenerateChunk:
    """Streaming chunk from a generation endpoint."""

    delta: str
    model: str
    done: bool
