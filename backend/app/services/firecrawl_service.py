"""Website scraping service powered by Firecrawl."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from firecrawl import FirecrawlApp

logger = logging.getLogger(__name__)


class FirecrawlService:
    """Thin async wrapper around the synchronous Firecrawl SDK."""

    def __init__(self, api_key: str) -> None:
        self._app = FirecrawlApp(api_key=api_key)

    # ── helpers ─────────────────────────────────────────────────────────

    def _scrape_url_compat(self, url: str, **kwargs: Any) -> Any:
        """Call Firecrawl scrape_url across SDK versions."""
        try:
            return self._app.scrape_url(url, **kwargs)
        except TypeError:
            params: dict[str, Any] = {}
            for key, value in kwargs.items():
                if value is None:
                    continue
                if key == "only_main_content":
                    params["onlyMainContent"] = value
                else:
                    params[key] = value
            return self._app.scrape_url(url, params=params)

    def _scrape_sync(self, url: str) -> dict[str, Any]:
        """Run the blocking Firecrawl call (executed in a thread)."""
        return self._scrape_url_compat(
            url,
            formats=["markdown", "html", "screenshot"],
            only_main_content=False,
        )

    def _map_url_sync(self, url: str) -> Any:
        """Run the blocking Firecrawl map call (executed in a thread)."""
        return self._app.map_url(url)

    async def scrape_url(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Scrape a page with SDK-version-compatible argument handling."""
        raw = await asyncio.to_thread(self._scrape_url_compat, url, **kwargs)
        if isinstance(raw, dict):
            return raw
        return raw.__dict__ if hasattr(raw, "__dict__") else {}

    async def map_url(self, url: str) -> Any:
        """Map a site using the blocking Firecrawl client."""
        return await asyncio.to_thread(self._map_url_sync, url)

    # ── public API ──────────────────────────────────────────────────────

    async def scrape_website(self, url: str) -> dict[str, Any]:
        """Scrape *url* and return markdown content plus any branding data.

        Returns
        -------
        dict
            ``{"markdown": str, "branding": dict}`` where *branding* may
            contain ``logo``, ``colors``, and ``fonts`` when available.
        """
        result: dict[str, Any] = {"markdown": "", "branding": {}}

        try:
            raw = await asyncio.to_thread(self._scrape_sync, url)

            # firecrawl-py returns a dict (or ScrapeResponse dataclass).
            if isinstance(raw, dict):
                data = raw
            else:
                # ScrapeResponse — convert to dict for uniform access.
                data = raw.__dict__ if hasattr(raw, "__dict__") else {}

            result["markdown"] = (
                data.get("markdown", "")
                or data.get("content", "")
                or ""
            )

            # Extract branding from the dedicated branding format when available.
            branding: dict[str, Any] = {}
            if data.get("branding"):
                branding = data["branding"]
            else:
                # Fall back to metadata hints.
                metadata = data.get("metadata", {}) or {}
                if metadata.get("og:image"):
                    branding["logo"] = metadata["og:image"]
                if metadata.get("theme-color"):
                    branding["colors"] = {"primary": metadata["theme-color"]}

            result["branding"] = branding

        except Exception:
            logger.exception("Firecrawl scrape failed for %s", url)
            # Return partial / empty result so the pipeline can continue
            # with whatever the LLM can infer from limited data.

        return result
