# babellm - Lightweight Ollama Client Plan                                                                                                                     
                                                                                                                                                            
## Context                                                                                                                                                      
                                                                                                                                                            
Build a lightweight, modular Python LLM client library starting with Ollama as the first provider. 
The library must be extensible to other providers (OpenAI, Anthropic, etc.) in the future with zero breaking changes. 
Supports chat, text generation, embeddings, and streaming — all with both sync and async APIs.


## Directory Structure

babellm/
├── pyproject.toml
├── babellm/
│   ├── __init__.py
│   ├── types.py
│   ├── exceptions.py            # Exception hierarchy
│   ├── _http.py                 # Thin httpx wrapper (sync + async + streaming)
│   ├── base/
│   │   ├── __init__.py
│   │   └── provider.py 
│   └── providers/
│       └── ollama/
│           ├── __init__.py
│           ├── client.py        
│           └── _serializers.py 
└── tests/
    ├── conftest.py
    ├── test_types.py
    └── test_ollama_client.py


## Implementation Order

1. pyproject.toml

Single runtime dependency: httpx>=0.27. Package managed with uv. Linting and formatting with ruff.

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "babellm"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = ["httpx>=0.27"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "respx>=0.21", "ruff>=0.9"]

[tool.ruff]
target-version = "py314"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.ruff.format]
# ruff format replaces black

[tool.pytest.ini_options]
asyncio_mode = "auto"

uv commands:
uv init --no-workspace         # if not already initialized
uv add httpx
uv add --dev pytest pytest-asyncio respx ruff
uv run pytest                  # run tests
uv run ruff check .            # lint
uv run ruff format .           # format

2. babellm/types.py

Stdlib dataclasses only — no pydantic, no weight. All output types shared across providers.

Key types:
- Role(str, Enum) — SYSTEM, USER, ASSISTANT
- Message(role, content)
- Usage(prompt_tokens, completion_tokens, total_tokens)
- ChatResponse(message, model, done, usage, raw)
- GenerateResponse(text, model, done, usage, raw)
- EmbeddingResponse(embedding, model, raw) — embedding: list[float]
- ChatChunk(delta, model, done) — streaming
- GenerateChunk(delta, model, done) — streaming

3. babellm/exceptions.py

BabelLMError
├── ProviderError(status_code, raw)
│   ├── AuthenticationError
│   ├── RateLimitError
│   └── ModelNotFoundError
├── ConnectionError
└── StreamingError

4. babellm/_http.py

Wraps httpx.Client and httpx.AsyncClient. Exposes:
- post(path, json) -> dict
- post_stream(path, json) -> Iterator[dict]  — NDJSON line-by-line
- apost(path, json) -> dict
- apost_stream(path, json) -> AsyncIterator[dict]
- Context manager support (sync + async)

Streaming uses httpx's stream() + iter_lines() + json.loads() per line.

5. babellm/base/provider.py

Abstract base class with abc.ABC. Defines the full contract:

Sync: chat, chat_stream, generate, generate_stream, embed
Async: achat, achat_stream, agenerate, agenerate_stream, aembed

All methods accept **kwargs for provider-specific pass-through options.

6. babellm/providers/ollama/_serializers.py

Pure functions (no I/O, easily unit-tested):
- messages_to_ollama(messages) -> list[dict]
- ollama_to_chat_response(raw, model) -> ChatResponse
- ollama_to_chat_chunk(raw, model) -> ChatChunk
- ollama_to_generate_response(raw, model) -> GenerateResponse
- ollama_to_generate_chunk(raw, model) -> GenerateChunk
- ollama_to_embedding_response(raw, model) -> EmbeddingResponse
- ollama_to_usage(raw) -> Usage

Ollama endpoints:
- POST /api/chat — chat (stream=false/true)
- POST /api/generate — generate (stream=false/true)
- POST /api/embed — embeddings

7. babellm/providers/ollama/client.py

OllamaProvider(BaseLLMProvider). Defaults to http://localhost:11434.
- Delegates HTTP to HTTPClient
- Delegates serialization to _serializers
- Supports context managers (delegates to HTTPClient)

8. babellm/__init__.py

from .providers.ollama.client import OllamaProvider
from .types import Message, Role, ChatResponse, GenerateResponse, EmbeddingResponse, Usage
from .exceptions import BabelLMError, ProviderError

Example Usage

from babellm import OllamaProvider, Message, Role

# Sync chat
client = OllamaProvider()
response = client.chat(
    messages=[Message(role=Role.USER, content="What is 2+2?")],
    model="llama3.2",
)
print(response.message.content)

# Streaming
with OllamaProvider() as client:
    for chunk in client.chat_stream(messages=[...], model="llama3.2"):
        print(chunk.delta, end="", flush=True)

# Embeddings
result = client.embed(input="Hello world", model="nomic-embed-text")
vector: list[float] = result.embedding

# Async
async with OllamaProvider() as client:
    async for chunk in client.achat_stream(messages=[...], model="llama3.2"):
        print(chunk.delta, end="", flush=True)

Extending for Future Providers

1. Create babellm/providers/openai/ with client.py + _serializers.py
2. OpenAIProvider(BaseLLMProvider) — same structure, different base_url and auth header
3. Register in __init__.py
4. Zero changes to types.py, exceptions.py, base/provider.py, _http.py

Future: add babellm/factory.py with create_client(provider, **kwargs) when ≥2 providers exist.

### Verification

1. pip install -e ".[dev]"
2. Start ollama locally: ollama serve
3. Run tests: pytest tests/
4. Manual smoke test: run the example usage patterns above