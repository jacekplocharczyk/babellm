"""Text generation (single-shot completion) example."""

from babellm import OllamaProvider

provider = OllamaProvider(base_url="http://localhost:11434")

# Generate without conversation history
prompt = "The meaning of life is"

response = provider.generate(
    prompt=prompt,
    model="llama3.2:1b",
    system="You are a wise philosopher.",
    temperature=0.7,
    max_tokens=100
)

print("Generated text:")
print(response.text)
print(f"\nModel: {response.model}")
print(f"Done: {response.done}")

provider.close()
