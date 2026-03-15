"""Thin wrapper around httpx for sync and async HTTP operations."""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .exceptions import ConnectionError, ProviderError


class HTTPClient:
    """Manages httpx.Client and httpx.AsyncClient for HTTP operations."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._headers = headers or {}
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create sync client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self.timeout,
            )
        return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self.timeout,
            )
        return self._async_client

    def post(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        """POST JSON and return parsed response."""
        try:
            client = self._get_client()
            response = client.post(path, json=json_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self._raise_provider_error(e)
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}") from e

    def post_stream(
        self, path: str, json_data: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        """POST JSON and stream NDJSON response."""
        try:
            client = self._get_client()
            with client.stream("POST", path, json=json_data) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        yield json.loads(line)
        except httpx.HTTPStatusError as e:
            self._raise_provider_error(e)
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}") from e

    async def apost(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        """Async POST JSON and return parsed response."""
        try:
            client = self._get_async_client()
            response = await client.post(path, json=json_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self._raise_provider_error(e)
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}") from e

    async def apost_stream(
        self, path: str, json_data: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Async POST JSON and stream NDJSON response."""
        try:
            client = self._get_async_client()
            async with client.stream("POST", path, json=json_data) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield json.loads(line)
        except httpx.HTTPStatusError as e:
            self._raise_provider_error(e)
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}") from e

    def close(self) -> None:
        """Close sync client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close async client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> HTTPClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    async def __aenter__(self) -> HTTPClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.aclose()

    @staticmethod
    def _raise_provider_error(error: httpx.HTTPStatusError) -> None:
        """Convert httpx errors to babellm exceptions."""
        status = error.response.status_code
        try:
            raw = error.response.json()
        except Exception:
            try:
                text = error.response.text
            except Exception:
                text = str(error)
            raw = {"message": text}

        if status == 401:
            from .exceptions import AuthenticationError

            raise AuthenticationError(
                str(error), status_code=status, raw=raw
            ) from error
        if status == 403:
            from .exceptions import AuthenticationError

            raise AuthenticationError(
                str(error), status_code=status, raw=raw
            ) from error
        if status == 404:
            from .exceptions import ModelNotFoundError

            raise ModelNotFoundError(str(error), status_code=status, raw=raw) from error
        if status == 429:
            from .exceptions import RateLimitError

            raise RateLimitError(str(error), status_code=status, raw=raw) from error

        raise ProviderError(str(error), status_code=status, raw=raw) from error
