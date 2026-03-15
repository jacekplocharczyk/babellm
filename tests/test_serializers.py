"""Tests for Ollama serializers."""

from babellm.providers.ollama._serializers import (
    messages_to_ollama,
    ollama_to_chat_chunk,
    ollama_to_chat_response,
    ollama_to_embedding_response,
    ollama_to_generate_chunk,
    ollama_to_generate_response,
    ollama_to_usage,
)
from babellm.types import Message, Role


class TestMessagesToOllama:
    """Test message serialization."""

    def test_single_user_message(self):
        """Test converting a single user message."""
        messages = [Message(role=Role.USER, content="Hello")]
        result = messages_to_ollama(messages)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_multiple_messages(self):
        """Test converting multiple messages."""
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="What is 2+2?"),
            Message(role=Role.ASSISTANT, content="4"),
        ]
        result = messages_to_ollama(messages)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_all_roles(self):
        """Test all role types."""
        messages = [
            Message(role=Role.SYSTEM, content="system"),
            Message(role=Role.USER, content="user"),
            Message(role=Role.ASSISTANT, content="assistant"),
            Message(role=Role.TOOL, content="tool"),
        ]
        result = messages_to_ollama(messages)
        assert len(result) == 4
        assert [r["role"] for r in result] == ["system", "user", "assistant", "tool"]


class TestOllamaToUsage:
    """Test usage parsing."""

    def test_usage_from_response(self):
        """Test parsing usage from Ollama response."""
        raw = {
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        usage = ollama_to_usage(raw)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_with_zeros(self):
        """Test usage with zero counts."""
        raw = {"prompt_eval_count": 0, "eval_count": 0}
        usage = ollama_to_usage(raw)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_missing_fields(self):
        """Test usage with missing fields."""
        raw = {}
        usage = ollama_to_usage(raw)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestOllamaToChatResponse:
    """Test chat response parsing."""

    def test_basic_chat_response(self):
        """Test parsing a basic chat response."""
        raw = {
            "message": {"role": "assistant", "content": "Hello!"},
            "model": "llama3.2",
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        response = ollama_to_chat_response(raw, "llama3.2")
        assert response.message.role == Role.ASSISTANT
        assert response.message.content == "Hello!"
        assert response.model == "llama3.2"
        assert response.done is True
        assert response.usage is not None
        assert response.usage.prompt_tokens == 5

    def test_chat_response_without_usage(self):
        """Test chat response without token counts."""
        raw = {
            "message": {"role": "assistant", "content": "Response"},
            "model": "test",
            "done": False,
        }
        response = ollama_to_chat_response(raw, "test")
        assert response.message.content == "Response"
        assert response.usage is not None

    def test_chat_response_stores_raw(self):
        """Test that raw response is stored."""
        raw = {"message": {"role": "assistant", "content": "Test"}, "done": True}
        response = ollama_to_chat_response(raw, "test")
        assert response.raw == raw


class TestOllamaToChatChunk:
    """Test streaming chat chunk parsing."""

    def test_chat_chunk_parsing(self):
        """Test parsing a chat chunk."""
        raw = {
            "message": {"content": "Hello "},
            "model": "llama3.2",
            "done": False,
        }
        chunk = ollama_to_chat_chunk(raw, "llama3.2")
        assert chunk.delta == "Hello "
        assert chunk.model == "llama3.2"
        assert chunk.done is False

    def test_chat_chunk_done(self):
        """Test chat chunk with done flag."""
        raw = {"message": {"content": ""}, "model": "test", "done": True}
        chunk = ollama_to_chat_chunk(raw, "test")
        assert chunk.done is True

    def test_chat_chunk_empty_content(self):
        """Test chat chunk with empty content."""
        raw = {"message": {"content": ""}, "done": False}
        chunk = ollama_to_chat_chunk(raw, "test")
        assert chunk.delta == ""


class TestOllamaToGenerateResponse:
    """Test generate response parsing."""

    def test_basic_generate_response(self):
        """Test parsing a generate response."""
        raw = {
            "response": "def fibonacci(n):\n    return n",
            "model": "codellama",
            "done": True,
            "prompt_eval_count": 3,
            "eval_count": 15,
        }
        response = ollama_to_generate_response(raw, "codellama")
        assert response.text == "def fibonacci(n):\n    return n"
        assert response.model == "codellama"
        assert response.done is True
        assert response.usage.prompt_tokens == 3
        assert response.usage.completion_tokens == 15

    def test_generate_response_without_usage(self):
        """Test generate response without token counts."""
        raw = {
            "response": "Generated text",
            "model": "test",
            "done": False,
        }
        response = ollama_to_generate_response(raw, "test")
        assert response.text == "Generated text"
        assert response.usage is not None


class TestOllamaToGenerateChunk:
    """Test streaming generate chunk parsing."""

    def test_generate_chunk_parsing(self):
        """Test parsing a generate chunk."""
        raw = {"response": "def ", "model": "codellama", "done": False}
        chunk = ollama_to_generate_chunk(raw, "codellama")
        assert chunk.delta == "def "
        assert chunk.model == "codellama"
        assert chunk.done is False

    def test_generate_chunk_done(self):
        """Test generate chunk with done flag."""
        raw = {"response": "", "model": "test", "done": True}
        chunk = ollama_to_generate_chunk(raw, "test")
        assert chunk.done is True


class TestOllamaToEmbeddingResponse:
    """Test embedding response parsing."""

    def test_basic_embedding_response(self):
        """Test parsing an embedding response."""
        embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        raw = {
            "embedding": embedding_vector,
            "model": "nomic-embed-text",
        }
        response = ollama_to_embedding_response(raw, "nomic-embed-text")
        assert response.embedding == embedding_vector
        assert response.model == "nomic-embed-text"

    def test_embedding_response_stores_raw(self):
        """Test that raw response is stored."""
        raw = {"embedding": [0.1, 0.2], "model": "test"}
        response = ollama_to_embedding_response(raw, "test")
        assert response.raw == raw

    def test_embedding_response_empty_vector(self):
        """Test embedding with empty vector."""
        raw = {"embedding": [], "model": "test"}
        response = ollama_to_embedding_response(raw, "test")
        assert response.embedding == []
