"""Streaming chat example."""

from babellm import Message, OllamaProvider, Role

provider = OllamaProvider(base_url="http://localhost:11434")

# Stream response token by token
message = Message(
    role=Role.USER,
    content="Tell me a short story about a robot learning to paint"
)

print("Streaming response:")
for chunk in provider.chat_stream(messages=[message], model="llama3.2:1b"):
    print(chunk.delta, end="", flush=True)
    if chunk.done:
        print("\n[Stream complete]")
        break

provider.close()
