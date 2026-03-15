"""Asynchronous API usage example."""

import asyncio

from babellm import Message, OllamaProvider, Role


async def main():
    provider = OllamaProvider(base_url="http://localhost:11434")

    # Async chat
    message = Message(role=Role.USER, content="Explain quantum computing briefly")
    response = await provider.achat(messages=[message], model="llama3.2:1b")

    print("Async chat response:")
    print(response.message.content)

    # Async streaming
    print("\nAsync streaming:")
    async for chunk in provider.achat_stream(
        messages=[Message(role=Role.USER, content="Write a haiku about coding")],
        model="llama3.2:1b"
    ):
        print(chunk.delta, end="", flush=True)
        if chunk.done:
            print("\n[Stream complete]")
            break

    # Async generate
    print("\nAsync generate:")
    gen_response = await provider.agenerate(
        prompt="To be or not to be",
        model="llama3.2:1b"
    )
    print(gen_response.text)

    await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
