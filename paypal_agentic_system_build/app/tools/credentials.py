"""Credential layer.

The only component that knows secrets. Tokens are fetched, cached and refreshed
here; nothing above this layer (agent, planner, MCP, registry, logs) ever sees
them.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import httpx

from app.config import Settings, get_settings

TOKEN_PATH = "/v1/oauth2/token"
EXPIRY_SKEW_SECONDS = 60


class CredentialError(RuntimeError):
    """Raised when credentials are missing or rejected by the provider."""


@dataclass
class _CachedToken:
    value: str
    expires_at: float

    def valid(self) -> bool:
        return bool(self.value) and time.time() < self.expires_at - EXPIRY_SKEW_SECONDS


@dataclass
class CredentialManager:
    """Fetches and caches an OAuth2 client-credentials access token."""

    settings: Settings = field(default_factory=get_settings)
    client: httpx.AsyncClient | None = None
    _token: _CachedToken | None = None

    def basic_auth_header(self) -> str:
        if not self.settings.paypal_client_id or not self.settings.paypal_client_secret:
            raise CredentialError("PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET are not configured")
        raw = f"{self.settings.paypal_client_id}:{self.settings.paypal_client_secret}"
        return "Basic " + base64.b64encode(raw.encode("utf-8")).decode("ascii")

    def invalidate(self) -> None:
        """Drop the cached token so the next call re-authenticates."""
        self._token = None

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._token and self._token.valid():
            return self._token.value
        return await self._fetch_token()

    async def _fetch_token(self) -> str:
        client = self.client or httpx.AsyncClient(timeout=20.0)
        owns_client = self.client is None
        try:
            response = await client.post(
                f"{self.settings.paypal_base_url.rstrip('/')}{TOKEN_PATH}",
                headers={
                    "Authorization": self.basic_auth_header(),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                content="grant_type=client_credentials",
            )
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code != 200:
            raise CredentialError(f"token request failed with HTTP {response.status_code}")

        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise CredentialError("token response did not contain an access_token")
        self._token = _CachedToken(
            value=token, expires_at=time.time() + float(payload.get("expires_in", 300))
        )
        return token

    async def authorization_header(self, auth_type: str, *, force_refresh: bool = False) -> str:
        if auth_type == "basic":
            return self.basic_auth_header()
        return f"Bearer {await self.get_access_token(force_refresh=force_refresh)}"
