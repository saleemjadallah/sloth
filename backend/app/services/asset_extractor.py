"""Extract, download, and classify visual assets from a website.

This service:
1. Uses Firecrawl to crawl multiple pages and discover all images
2. Filters for high-quality, ad-usable images (skips tiny icons, tracking pixels, etc.)
3. Downloads assets to local storage (or R2 in production)
4. Uses an LLM to classify and describe each asset for ad creation
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.services.firecrawl_service import FirecrawlService

logger = logging.getLogger(__name__)

# Minimum dimensions for an image to be considered usable for ads
MIN_USABLE_WIDTH = 200
MIN_USABLE_HEIGHT = 200
# Skip common non-content image patterns
SKIP_PATTERNS = re.compile(
    r"(tracking|pixel|spacer|blank|1x1|beacon|analytics"
    r"|facebook\.com/tr|google-analytics|doubleclick"
    r"|\.svg$|\.ico$|favicon|spinner|loader|arrow|chevron"
    r"|data:image/(?:gif|png);base64,[\w+/=]{0,200}$)",  # tiny inline data URIs
    re.IGNORECASE,
)
# File extensions we care about
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp", ".tiff"}


class AssetExtractor:
    """Discovers, downloads, and prepares website images for ad creation."""

    def __init__(
        self,
        firecrawl_service: FirecrawlService,
        storage_dir: str = "assets",
    ) -> None:
        self._firecrawl = firecrawl_service
        self._storage_base = Path(storage_dir)
        self._storage_base.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    async def extract_assets(
        self,
        website_url: str,
        brand_id: str,
        max_pages: int = 10,
    ) -> list[dict[str, Any]]:
        """Crawl a website, discover images, download and prepare them.

        Returns a list of asset dicts ready to be persisted as BrandAsset records.
        """
        # 1. Discover all image URLs across the site
        image_entries = await self._discover_images(website_url, max_pages)
        logger.info("Discovered %d candidate images from %s", len(image_entries), website_url)

        # 2. Filter obvious junk
        filtered = [e for e in image_entries if self._is_candidate(e["url"])]
        logger.info("After filtering: %d candidate images", len(filtered))

        # 3. Download images concurrently (with limit)
        brand_dir = self._storage_base / brand_id
        brand_dir.mkdir(parents=True, exist_ok=True)

        downloaded = await self._download_batch(filtered, brand_dir)
        logger.info("Downloaded %d images", len(downloaded))

        return downloaded

    # ── Image Discovery ───────────────────────────────────────────────────

    async def _discover_images(
        self,
        website_url: str,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        """Use Firecrawl to crawl the site and extract all image URLs."""
        # First, try to get a site map of URLs
        pages_to_scrape = await self._get_site_pages(website_url, max_pages)

        all_images: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        # Scrape each page for images
        tasks = [
            self._extract_images_from_page(page_url)
            for page_url in pages_to_scrape
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for page_url, result in zip(pages_to_scrape, results):
            if isinstance(result, Exception):
                logger.warning("Failed to extract images from %s: %s", page_url, result)
                continue
            for img in result:
                # Normalize and deduplicate
                abs_url = urljoin(page_url, img["url"])
                if abs_url not in seen_urls:
                    seen_urls.add(abs_url)
                    img["url"] = abs_url
                    img["source_page"] = page_url
                    all_images.append(img)

        return all_images

    async def _get_site_pages(self, website_url: str, max_pages: int) -> list[str]:
        """Get a list of important pages to scrape from the website."""
        try:
            # Use Firecrawl map to discover pages
            urls = await asyncio.to_thread(
                self._firecrawl._app.map_url, website_url
            )
            if isinstance(urls, dict):
                urls = urls.get("links", [])
            if isinstance(urls, list) and urls:
                # Prioritize key pages
                prioritized = self._prioritize_pages(urls, website_url)
                return prioritized[:max_pages]
        except Exception:
            logger.warning("Firecrawl map failed, falling back to homepage + common paths")

        # Fallback: homepage + common important pages
        parsed = urlparse(website_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        common_paths = [
            "", "/about", "/products", "/services", "/shop",
            "/gallery", "/portfolio", "/team", "/pricing",
        ]
        return [f"{base}{p}" for p in common_paths[:max_pages]]

    def _prioritize_pages(self, urls: list[str], website_url: str) -> list[str]:
        """Sort pages by likely asset richness — product/about/gallery pages first."""
        parsed_base = urlparse(website_url)
        same_domain = [
            u for u in urls
            if urlparse(u).netloc == parsed_base.netloc
        ]

        high_priority_keywords = [
            "product", "shop", "store", "gallery", "portfolio",
            "about", "team", "service", "collection", "catalog",
            "work", "case-stud", "testimonial", "review", "pricing",
        ]
        medium_priority_keywords = ["blog", "news", "article", "post"]

        def score(url: str) -> int:
            lower = url.lower()
            if lower.rstrip("/") == website_url.rstrip("/"):
                return 100  # Homepage always first
            for kw in high_priority_keywords:
                if kw in lower:
                    return 50
            for kw in medium_priority_keywords:
                if kw in lower:
                    return 10
            return 1

        same_domain.sort(key=score, reverse=True)
        return same_domain

    async def _extract_images_from_page(self, page_url: str) -> list[dict[str, Any]]:
        """Scrape a single page and extract all image references."""
        try:
            raw = await asyncio.to_thread(
                self._firecrawl._app.scrape_url,
                page_url,
                {"formats": ["markdown", "links"]},
            )
            if not isinstance(raw, dict):
                raw = raw.__dict__ if hasattr(raw, "__dict__") else {}
        except Exception:
            logger.warning("Failed to scrape %s for images", page_url)
            return []

        images: list[dict[str, Any]] = []
        markdown = raw.get("markdown", "") or ""
        metadata = raw.get("metadata", {}) or {}

        # Extract image URLs from markdown (![alt](url) pattern)
        md_images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", markdown)
        for alt_text, img_url in md_images:
            images.append({
                "url": img_url,
                "alt_text": alt_text or None,
                "context": self._get_surrounding_text(markdown, img_url),
            })

        # Also grab OG image, twitter image, etc. from metadata
        for meta_key in ["og:image", "twitter:image", "image"]:
            if metadata.get(meta_key):
                img_url = metadata[meta_key]
                if img_url and not any(i["url"] == img_url for i in images):
                    images.append({
                        "url": img_url,
                        "alt_text": metadata.get("og:image:alt"),
                        "context": metadata.get("og:description") or metadata.get("description"),
                    })

        # Extract links that look like images
        links = raw.get("links", []) or []
        for link in links:
            link_url = link if isinstance(link, str) else link.get("url", "")
            parsed = urlparse(link_url)
            ext = Path(parsed.path).suffix.lower()
            if ext in IMAGE_EXTENSIONS and not any(i["url"] == link_url for i in images):
                images.append({
                    "url": link_url,
                    "alt_text": None,
                    "context": None,
                })

        return images

    # ── Filtering ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_candidate(url: str) -> bool:
        """Check if a URL looks like a usable content image (not junk)."""
        if SKIP_PATTERNS.search(url):
            return False
        # Check extension if present
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()
        if ext and ext not in IMAGE_EXTENSIONS:
            return False
        # Skip very short data URIs (likely 1px tracking pixels)
        if url.startswith("data:") and len(url) < 500:
            return False
        return True

    # ── Download ──────────────────────────────────────────────────────────

    async def _download_batch(
        self,
        entries: list[dict[str, Any]],
        brand_dir: Path,
        concurrency: int = 10,
    ) -> list[dict[str, Any]]:
        """Download images concurrently with a semaphore."""
        sem = asyncio.Semaphore(concurrency)
        results: list[dict[str, Any]] = []

        async def _download_one(entry: dict[str, Any]) -> dict[str, Any] | None:
            async with sem:
                return await self._download_image(entry, brand_dir)

        tasks = [_download_one(e) for e in entries]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in raw_results:
            if isinstance(r, Exception):
                continue
            if r is not None:
                results.append(r)

        return results

    async def _download_image(
        self,
        entry: dict[str, Any],
        brand_dir: Path,
    ) -> dict[str, Any] | None:
        """Download a single image and return its metadata."""
        url = entry["url"]

        # Skip data URIs
        if url.startswith("data:"):
            return None

        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=15.0
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None

            data = resp.content
            if len(data) < 1000:  # Skip tiny images (<1KB)
                return None

            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = mimetypes.guess_extension(content_type.split(";")[0]) or ".jpg"
            file_name = f"{url_hash}{ext}"
            file_path = brand_dir / file_name

            # Write to disk
            file_path.write_bytes(data)

            # Try to get dimensions
            width, height = self._get_image_dimensions(data)

            # Skip too-small images
            if width and height and (width < MIN_USABLE_WIDTH or height < MIN_USABLE_HEIGHT):
                file_path.unlink(missing_ok=True)
                return None

            return {
                "source_url": url,
                "source_page": entry.get("source_page"),
                "stored_url": str(file_path),
                "file_name": file_name,
                "file_size": len(data),
                "mime_type": content_type.split(";")[0],
                "width": width,
                "height": height,
                "alt_text": entry.get("alt_text"),
                "context": entry.get("context"),
            }

        except Exception:
            logger.debug("Failed to download %s", url)
            return None

    @staticmethod
    def _get_image_dimensions(data: bytes) -> tuple[int | None, int | None]:
        """Extract image dimensions without PIL if possible, else try PIL."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            return img.size
        except Exception:
            pass

        # Fallback: parse PNG header
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return w, h

        # JPEG: scan for SOF marker
        if data[:2] == b"\xff\xd8":
            i = 2
            while i < len(data) - 9:
                if data[i] != 0xFF:
                    break
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    h = int.from_bytes(data[i + 5 : i + 7], "big")
                    w = int.from_bytes(data[i + 7 : i + 9], "big")
                    return w, h
                seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
                i += 2 + seg_len

        return None, None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _get_surrounding_text(markdown: str, img_url: str, chars: int = 200) -> str | None:
        """Extract text surrounding an image reference in the markdown."""
        idx = markdown.find(img_url)
        if idx < 0:
            return None
        start = max(0, idx - chars)
        end = min(len(markdown), idx + len(img_url) + chars)
        snippet = markdown[start:end]
        # Clean up markdown artifacts
        snippet = re.sub(r"!\[[^\]]*\]\([^)]*\)", "[image]", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        return snippet if snippet else None
