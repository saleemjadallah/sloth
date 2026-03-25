"""Late.dev (currently docs-redirected via Zernio) publishing service."""

from __future__ import annotations

from typing import Any

import httpx


class LateService:
    """Thin async client around the Late social publishing API."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def list_accounts(
        self,
        *,
        profile_id: str | None = None,
        platform: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if profile_id:
            params["profileId"] = profile_id
        if platform:
            params["platform"] = platform

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self._base_url}/accounts",
                headers=self._headers,
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("accounts", [])

    async def create_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self._base_url}/posts",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()
