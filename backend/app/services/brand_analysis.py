"""Orchestrates the full brand-analysis pipeline.

1. Scrape the website via Firecrawl.
2. Analyse the scraped content via Claude.
3. Merge visual branding data with LLM-derived insights.
4. Crawl site to extract all usable visual assets.
5. Classify assets with LLM vision for ad-creation use.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.asset_classifier import AssetClassifier
from app.services.asset_extractor import AssetExtractor
from app.services.firecrawl_service import FirecrawlService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class BrandAnalysisService:
    """High-level service that composes scraping + LLM analysis + asset extraction."""

    def __init__(
        self,
        firecrawl_service: FirecrawlService,
        llm_service: LLMService,
        asset_extractor: AssetExtractor,
        asset_classifier: AssetClassifier,
    ) -> None:
        self._firecrawl = firecrawl_service
        self._llm = llm_service
        self._asset_extractor = asset_extractor
        self._asset_classifier = asset_classifier

    # ── public API ──────────────────────────────────────────────────────

    async def analyze(
        self,
        website_url: str,
        brand_id: str = "",
    ) -> dict[str, Any]:
        """Run the end-to-end brand analysis and return a merged profile.

        Steps run concurrently where possible:
        - Scrape homepage → LLM brand analysis (sequential)
        - Asset extraction (concurrent with brand analysis)
        """
        errors: list[str] = []

        # 1. Scrape homepage for brand content
        try:
            scrape_result = await self._firecrawl.scrape_website(website_url)
        except Exception as exc:
            logger.exception("Scrape step failed for %s", website_url)
            errors.append(f"Scrape failed: {exc}")
            scrape_result = {"markdown": "", "branding": {}}

        markdown = scrape_result.get("markdown", "")
        branding = scrape_result.get("branding", {})

        if not markdown:
            errors.append("No markdown content was extracted from the website.")

        # 2. Run LLM analysis and asset extraction concurrently
        llm_task = self._run_llm_analysis(website_url, markdown, branding, errors)
        asset_task = self._run_asset_extraction(website_url, brand_id, errors)

        llm_analysis, raw_assets = await asyncio.gather(llm_task, asset_task)

        # 3. Classify assets with LLM vision (uses brand context from step 2)
        brand_context = self._build_brand_context(llm_analysis)
        classified_assets: list[dict[str, Any]] = []
        if raw_assets:
            try:
                classified_assets = await self._asset_classifier.classify_assets(
                    assets=raw_assets,
                    website_url=website_url,
                    brand_context=brand_context,
                )
            except Exception as exc:
                logger.exception("Asset classification failed for %s", website_url)
                errors.append(f"Asset classification failed: {exc}")
                # Keep raw assets without classification
                classified_assets = raw_assets

        # 4. Merge everything
        profile = self._merge(branding, llm_analysis)
        profile["raw_analysis"] = llm_analysis
        profile["website_url"] = website_url
        profile["assets"] = classified_assets

        if errors:
            profile["errors"] = errors

        logger.info(
            "Brand analysis complete for %s: %d assets extracted (%d usable)",
            website_url,
            len(classified_assets),
            sum(1 for a in classified_assets if a.get("is_usable", True)),
        )

        return profile

    # ── internal pipeline steps ───────────────────────────────────────────

    async def _run_llm_analysis(
        self,
        website_url: str,
        markdown: str,
        branding: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any]:
        """Run LLM brand analysis, catching errors."""
        try:
            return await self._llm.analyze_brand(
                website_url=website_url,
                markdown_content=markdown,
                branding_data=branding,
            )
        except Exception as exc:
            logger.exception("LLM analysis failed for %s", website_url)
            errors.append(f"LLM analysis failed: {exc}")
            return {}

    async def _run_asset_extraction(
        self,
        website_url: str,
        brand_id: str,
        errors: list[str],
    ) -> list[dict[str, Any]]:
        """Run asset extraction, catching errors."""
        try:
            return await self._asset_extractor.extract_assets(
                website_url=website_url,
                brand_id=brand_id or "temp",
                max_pages=16,
            )
        except Exception as exc:
            logger.exception("Asset extraction failed for %s", website_url)
            errors.append(f"Asset extraction failed: {exc}")
            return []

    @staticmethod
    def _build_brand_context(llm_analysis: dict[str, Any]) -> str:
        """Build a short brand context string for the asset classifier."""
        parts = []
        if llm_analysis.get("company_name"):
            parts.append(f"Company: {llm_analysis['company_name']}")
        if llm_analysis.get("industry"):
            parts.append(f"Industry: {llm_analysis['industry']}")
        if llm_analysis.get("value_propositions"):
            vps = llm_analysis["value_propositions"][:3]
            parts.append(f"Value props: {', '.join(vps)}")
        products = llm_analysis.get("products", [])
        if products:
            names = [p.get("name", "") for p in products[:5] if p.get("name")]
            if names:
                parts.append(f"Products: {', '.join(names)}")
        return "; ".join(parts)

    # ── merge ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_string(value: Any) -> str | None:
        """Return the first non-empty string from a scalar or sequence."""
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, (list, tuple, set)):
            for item in value:
                picked = BrandAnalysisService._pick_string(item)
                if picked:
                    return picked
            return None
        return str(value)

    @classmethod
    def _normalize_colors(cls, colors: Any) -> dict[str, str] | None:
        """Normalize palette values to simple strings for API/storage safety."""
        if not isinstance(colors, dict):
            return None

        normalized = {
            key: picked
            for key in ("primary", "secondary", "accent")
            if (picked := cls._pick_string(colors.get(key)))
        }

        return normalized or None

    @classmethod
    def _normalize_fonts(cls, fonts: Any) -> dict[str, str] | None:
        """Normalize font values to simple strings for API/storage safety."""
        if not isinstance(fonts, dict):
            return None

        normalized = {
            key: picked
            for key in ("heading", "body")
            if (picked := cls._pick_string(fonts.get(key)))
        }

        return normalized or None

    @staticmethod
    def _merge(
        branding: dict[str, Any],
        llm: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine scraped branding data with LLM insights.

        Scraped visual data (colors, fonts, logo) takes precedence when
        present because it comes from the actual rendered page.
        """
        colors = (
            BrandAnalysisService._normalize_colors(branding.get("colors"))
            or BrandAnalysisService._normalize_colors(llm.get("colors"))
        )
        fonts = (
            BrandAnalysisService._normalize_fonts(branding.get("fonts"))
            or BrandAnalysisService._normalize_fonts(llm.get("fonts"))
        )
        logo_url = (
            BrandAnalysisService._pick_string(branding.get("logo"))
            or BrandAnalysisService._pick_string(llm.get("logo_url"))
        )

        return {
            "name": llm.get("company_name", ""),
            "logo_url": logo_url,
            "colors": colors,
            "fonts": fonts,
            "voice": llm.get("voice"),
            "value_propositions": llm.get("value_propositions"),
            "target_audience": llm.get("target_audience"),
            "products": llm.get("products"),
            "industry": llm.get("industry"),
        }
