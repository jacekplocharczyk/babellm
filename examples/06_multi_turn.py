"""Multi-turn conversation example."""

from babellm import Message, OllamaProvider, Role

provider = OllamaProvider(base_url="http://localhost:11434")

# Build a conversation with history
messages = [
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="What is Python?"),
]

# First turn
response1 = provider.chat(messages=messages, model="llama3.2:1b")
print("User: What is Python?")
print(f"Assistant: {response1.message.content}\n")

# Add assistant response to history and continue
messages.append(Message(role=Role.ASSISTANT, content=response1.message.content))
messages.append(Message(role=Role.USER, content="What can I use it for?"))

# Second turn
response2 = provider.chat(messages=messages, model="llama3.2:1b")
print("User: What can I use it for?")
print(f"Assistant: {response2.message.content}\n")

# Continue with more turns
messages.append(Message(role=Role.ASSISTANT, content=response2.message.content))
messages.append(Message(role=Role.USER, content="Show me a simple example"))

response3 = provider.chat(messages=messages, model="llama3.2:1b")
print("User: Show me a simple example")
print(f"Assistant: {response3.message.content}\n")

provider.close()
