"""Pytest configuration and shared fixtures."""

import pytest
import respx
from httpx import AsyncClient, Client


@pytest.fixture
def http_client():
    """Create a test HTTP client."""
    return Client(base_url="http://localhost:11434")


@pytest.fixture
async def http_async_client():
    """Create a test async HTTP client."""
    async with AsyncClient(base_url="http://localhost:11434") as client:
        yield client


@pytest.fixture
def mock_http():
    """Mock HTTP responses with respx."""
    with respx.mock:
        yield respx
