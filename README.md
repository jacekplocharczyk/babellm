# babellm

A lightweight, modular Python LLM client library. Start with Ollama, extend to any provider.

## Features

- **Lightweight**: Single runtime dependency (`httpx`)
- **Modular**: Clean provider abstraction for easy extensibility
- **Async-native**: Full async/await support alongside sync APIs
- **Streaming**: Built-in streaming for chat and generation
- **Type-safe**: Pure dataclasses, fully type-annotated
- **Zero framework bloat**: Stdlib-only (dataclasses, abc, enum)

## Installation

### For development:

```bash
uv sync
```

### For distribution (future):

```bash
pip install babellm
```

## Quick Start

### Sync Chat

```python
from babellm import OllamaProvider, Message, Role

client = OllamaProvider()  # defaults to http://localhost:11434

response = client.chat(
    messages=[
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What is 2+2?"),
    ],
    model="llama3.2",
)
print(response.message.content)
```

### Streaming Chat

```python
with OllamaProvider() as client:
    for chunk in client.chat_stream(
        messages=[Message(role=Role.USER, content="Tell me a joke.")],
        model="llama3.2",
    ):
        print(chunk.delta, end="", flush=True)
```

### Text Generation

```python
client = OllamaProvider()
response = client.generate(
    prompt="def fibonacci(n):",
    system="You are a Python code expert. Output code only.",
    model="codellama",
)
print(response.text)
```

### Embeddings

```python
result = client.embed(
    input=["Hello world", "Goodbye world"],
    model="nomic-embed-text",
)
# For batch embeddings, returns the first one
# Use custom handling for all embeddings if needed
vector: list[float] = result.embedding
```

### Async Operations

```python
import asyncio

async def main():
    async with OllamaProvider() as client:
        response = await client.achat(
            messages=[Message(role=Role.USER, content="Hello!")],
            model="llama3.2",
        )
        print(response.message.content)

asyncio.run(main())
```

### Streaming with Async

```python
async def stream_response():
    async with OllamaProvider() as client:
        async for chunk in client.achat_stream(
            messages=[Message(role=Role.USER, content="Explain quantum entanglement.")],
            model="llama3.2",
        ):
            print(chunk.delta, end="", flush=True)

asyncio.run(stream_response())
```

### Multi-turn Conversations

The library is stateless — you manage conversation history:

```python
history: list[Message] = []

client = OllamaProvider()

# User turn 1
history.append(Message(role=Role.USER, content="What's your name?"))
response = client.chat(messages=history, model="llama3.2")
history.append(response.message)
print(response.message.content)

# User turn 2
history.append(Message(role=Role.USER, content="What can you do?"))
response = client.chat(messages=history, model="llama3.2")
history.append(response.message)
print(response.message.content)
```

## API Reference

### OllamaProvider

```python
client = OllamaProvider(
    base_url="http://localhost:11434",
    timeout=120.0,
    headers=None,  # Custom headers dict
)
```

**Methods:**

- `chat(messages, model, *, temperature=None, max_tokens=None, stop=None, **kwargs)` → `ChatResponse`
- `chat_stream(messages, model, **kwargs)` → `Iterator[ChatChunk]`
- `generate(prompt, model, *, system=None, temperature=None, max_tokens=None, **kwargs)` → `GenerateResponse`
- `generate_stream(prompt, model, **kwargs)` → `Iterator[GenerateChunk]`
- `embed(input, model, **kwargs)` → `EmbeddingResponse`

**Async versions** (same signatures, prefixed with `a`):
- `achat`, `achat_stream`, `agenerate`, `agenerate_stream`, `aembed`

### Types

**Message**
```python
Message(role: Role, content: str)
```

**Role** (enum)
- `Role.SYSTEM`
- `Role.USER`
- `Role.ASSISTANT`
- `Role.TOOL` (reserved for future use)

**ChatResponse**
```python
ChatResponse(
    message: Message,
    model: str,
    done: bool,
    usage: Usage | None,
    raw: dict,
)
```

**GenerateResponse**
```python
GenerateResponse(
    text: str,
    model: str,
    done: bool,
    usage: Usage | None,
    raw: dict,
)
```

**EmbeddingResponse**
```python
EmbeddingResponse(
    embedding: list[float],
    model: str,
    raw: dict,
)
```

**Usage**
```python
Usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
)
```

## Exceptions

All exceptions inherit from `BabelLMError`:

- `ProviderError` — HTTP/provider error (status_code, raw response)
- `AuthenticationError` — Auth failed (401/403)
- `RateLimitError` — Rate limit exceeded (429)
- `ModelNotFoundError` — Model not found (404)
- `ConnectionError` — Network-level failure
- `StreamingError` — Error during streaming

```python
try:
    response = client.chat(...)
except AuthenticationError as e:
    print(f"Auth failed: {e.status_code}")
except RateLimitError:
    print("Rate limited, retry later")
except ProviderError as e:
    print(f"Provider error: {e.raw}")
```

## Configuration

### Custom Ollama Host

```python
client = OllamaProvider(base_url="http://192.168.1.100:11434")
```

### Custom Timeout

```python
client = OllamaProvider(timeout=300.0)  # 5 minutes
```

### Custom Headers

```python
client = OllamaProvider(
    headers={"X-Custom-Header": "value"}
)
```

### Provider-Specific Options

Pass via `**kwargs`:

```python
# Ollama supports keep_alive, top_p, top_k, etc.
response = client.chat(
    messages=[...],
    model="llama3.2",
    temperature=0.7,
    keep_alive="5m",  # Ollama-specific
)
```

## Extending for Other Providers

Adding OpenAI, Anthropic, or other providers requires only:

1. Create `babellm/providers/<provider>/client.py` implementing `BaseLLMProvider`
2. Create `babellm/providers/<provider>/_serializers.py` for API response parsing
3. Register in `babellm/__init__.py`

**Zero changes** to core types, exceptions, or HTTP layer.

## Development

### Install dependencies

```bash
uv sync
```

Or manually add:

```bash
uv add httpx
uv add --dev pytest pytest-asyncio respx ruff
```

### Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

### Run tests

```bash
uv run pytest tests/
```

## Architecture

```
babellm/
├── types.py           # Shared data types (Message, Response, etc.)
├── exceptions.py      # Exception hierarchy
├── _http.py           # httpx wrapper (handles streaming NDJSON)
├── base/provider.py   # Abstract BaseLLMProvider
└── providers/
    └── ollama/
        ├── client.py        # OllamaProvider
        └── _serializers.py  # Raw API dict → babellm types
```

**Design principles:**

- **Stateless**: No session management, caller owns conversation history
- **Provider-agnostic**: Single `BaseLLMProvider` interface for all providers
- **Lightweight**: httpx only, stdlib types (dataclasses, abc, enum)
- **Streaming-first**: NDJSON line-by-line parsing for all streaming endpoints
- **Type-safe**: Full type hints, no pydantic overhead

## License

See LICENSE file.
