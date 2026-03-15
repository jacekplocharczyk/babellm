# BabelLM Examples

Examples demonstrating how to use the BabelLM library with Ollama.

**Prerequisites**: Ollama server running on `localhost:11434`

## Examples

### 1. Basic Chat
[01_basic_chat.py](01_basic_chat.py)

Simple synchronous chat interaction. Shows how to:
- Create an OllamaProvider
- Send a single chat message
- Extract the response

```bash
python examples/01_basic_chat.py
```

### 2. Streaming Chat
[02_streaming_chat.py](02_streaming_chat.py)

Stream responses token-by-token. Shows how to:
- Use `chat_stream()` for real-time responses
- Process streaming chunks
- Handle completion signals

```bash
python examples/02_streaming_chat.py
```

### 3. Text Generation
[03_generate.py](03_generate.py)

Single-shot text generation (not conversational). Shows how to:
- Use the `generate()` endpoint
- Set system prompts
- Control generation parameters (temperature, max_tokens)

```bash
python examples/03_generate.py
```

### 4. Async Operations
[04_async_example.py](04_async_example.py)

Asynchronous API usage. Shows how to:
- Use async/await with `achat()` and `agenerate()`
- Stream async responses with `achat_stream()`
- Handle async context managers

```bash
python examples/04_async_example.py
```

### 5. Context Managers
[05_context_manager.py](05_context_manager.py)

Resource management with context managers. Shows how to:
- Use sync context manager (`with` statement)
- Use async context manager (`async with` statement)
- Automatic cleanup of HTTP connections

```bash
python examples/05_context_manager.py
```

### 6. Multi-turn Conversation
[06_multi_turn.py](06_multi_turn.py)

Build conversations with message history. Shows how to:
- Manage multiple message turns
- Keep conversation context
- Build chatbot-like interactions

```bash
python examples/06_multi_turn.py
```

## Running Examples

All examples assume Ollama is running with `llama3.2:1b`. If you don't have it, pull it:

```bash
ollama pull llama3.2:1b
```

Then run any example:
```bash
python examples/<example_name>.py
```

## API Overview

### Chat
```python
from babellm import OllamaProvider, Message, Role

provider = OllamaProvider()

# Sync
response = provider.chat(
    messages=[Message(role=Role.USER, content="Hello!")],
    model="llama3.2:1b"
)

# Streaming
for chunk in provider.chat_stream(
    messages=[Message(role=Role.USER, content="Hello!")],
    model="llama3.2:1b"
):
    print(chunk.delta, end="")
```

### Generate (Single-shot completion)
```python
# Sync
response = provider.generate(
    prompt="Complete this: The future of AI is",
    model="llama3.2:1b",
    system="You are a thoughtful AI researcher"
)

# Streaming
for chunk in provider.generate_stream(
    prompt="Complete this: The future of AI is",
    model="llama3.2:1b"
):
    print(chunk.delta, end="")
```

### Async API
```python
async def main():
    provider = OllamaProvider()
    response = await provider.achat(
        messages=[Message(role=Role.USER, content="Hello!")],
        model="llama3.2:1b"
    )
    await provider.aclose()

asyncio.run(main())
```

## Common Parameters

- **temperature**: Controls randomness (0.0 = deterministic, higher = more creative)
- **max_tokens**: Maximum length of generated text
- **stop**: List of stop sequences (generation stops when one is encountered)
- **model**: Model name (must be available in Ollama)

See provider documentation for additional provider-specific parameters.
