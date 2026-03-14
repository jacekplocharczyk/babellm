"""babellm - Lightweight LLM interface."""

from .exceptions import (
    AuthenticationError,
    BabelLMError,
    ConnectionError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    StreamingError,
)
from .providers.ollama import OllamaProvider
from .types import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    GenerateChunk,
    GenerateResponse,
    Message,
    Role,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    # Providers
    "OllamaProvider",
    # Types
    "Message",
    "Role",
    "ChatResponse",
    "ChatChunk",
    "GenerateResponse",
    "GenerateChunk",
    "EmbeddingResponse",
    "Usage",
    # Exceptions
    "BabelLMError",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "ConnectionError",
    "StreamingError",
]
