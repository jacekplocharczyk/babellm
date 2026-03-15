"""Basic synchronous chat example."""

from babellm import Message, OllamaProvider, Role

provider = OllamaProvider(base_url="http://localhost:11434")

# Single message
message = Message(role=Role.USER, content="What is the capital of France?")

response = provider.chat(messages=[message], model="llama3.2:1b")

print("Response:")
print(f"  Role: {response.message.role}")
print(f"  Content: {response.message.content}")
print(f"  Model: {response.model}")
print(f"  Done: {response.done}")

provider.close()
