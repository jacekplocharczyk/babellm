"""Context manager usage for automatic resource cleanup."""

import asyncio

from babellm import Message, OllamaProvider, Role


def sync_example():
    """Synchronous context manager example."""
    print("=== Sync Context Manager ===")

    with OllamaProvider(base_url="http://localhost:11434") as provider:
        message = Message(role=Role.USER, content="Hello from context manager!")
        response = provider.chat(messages=[message], model="llama3.2:1b")
        print(f"Response: {response.message.content}")
    # Provider automatically closed here

    print("Provider closed automatically\n")


async def async_example():
    """Asynchronous context manager example."""
    print("=== Async Context Manager ===")

    async with OllamaProvider(base_url="http://localhost:11434") as provider:
        message = Message(role=Role.USER, content="Hello from async context!")
        response = await provider.achat(messages=[message], model="llama3.2:1b")
        print(f"Response: {response.message.content}")
    # Provider automatically closed here

    print("Async provider closed automatically\n")


if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
