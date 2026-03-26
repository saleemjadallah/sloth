"""LLM-powered classification and description of extracted brand assets.

Uses Claude's vision capabilities to analyze each image and determine:
- Category (product, hero, lifestyle, logo, team, etc.)
- Description suitable for prompting video/image generation later
- Quality score for ad-creation suitability
- Searchable tags
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import anthropic

from app.services.asset_storage import AssetStorage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a brand asset analyst for an AI ad creation platform. Your job is to \
analyze images extracted from a company's website and classify them for use in \
creating advertising content.

For each image, determine:
1. **category**: one of: product, hero, lifestyle, logo, icon, team, testimonial, \
   banner, screenshot, infographic, food, fashion, interior, exterior, other
2. **description**: A detailed, prompt-ready description of what's in the image \
   (2-3 sentences). Describe subjects, composition, colors, mood, setting. This \
   description will later be used to generate similar content with AI video/image models.
3. **quality_score**: 1-10 rating for ad-creation suitability. Consider: \
   resolution, composition, subject clarity, brand-relevance, emotional appeal. \
   7+ means excellent ad material, 4-6 is usable, 1-3 is low quality or irrelevant.
4. **tags**: 5-10 searchable tags (lowercase, no hashtags)
5. **is_usable**: true if the image could realistically be used to create or \
   inspire an ad creative; false for generic stock, low-res, watermarked, or \
   irrelevant images.
6. **suggested_ad_use**: Brief suggestion for how this image could be used in ads \
   (e.g., "product showcase in carousel ad", "background for UGC overlay", \
   "hero image for brand story ad").

Always respond with a JSON object. No commentary outside the JSON."""

BATCH_USER_PROMPT = """\
Analyze the following {count} image(s) extracted from {website_url}.
Brand context: {brand_context}

For each image, return a JSON array with objects matching this schema:
[
  {{
    "index": 0,
    "category": "string",
    "description": "string",
    "quality_score": 1-10,
    "tags": ["string"],
    "is_usable": true/false,
    "suggested_ad_use": "string"
  }}
]

Analyze each image carefully. Focus on what makes each image valuable (or not) \
for creating advertising content for this brand."""


class AssetClassifier:
    """Classifies brand assets using Claude's vision capabilities."""

    def __init__(self, anthropic_api_key: str, storage: AssetStorage) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self._storage = storage

    async def classify_assets(
        self,
        assets: list[dict[str, Any]],
        website_url: str,
        brand_context: str = "",
        batch_size: int = 5,
    ) -> list[dict[str, Any]]:
        """Classify a list of downloaded assets using vision LLM.

        Processes in batches to stay within token limits while being efficient.
        """
        classified: list[dict[str, Any]] = []

        for i in range(0, len(assets), batch_size):
            batch = assets[i : i + batch_size]
            try:
                results = await self._classify_batch(
                    batch, website_url, brand_context
                )
                # Merge classification data back into asset dicts
                for asset, classification in zip(batch, results):
                    asset.update(classification)
                    classified.append(asset)
            except Exception:
                logger.exception(
                    "Failed to classify batch %d-%d", i, i + len(batch)
                )
                # Still include unclassified assets
                for asset in batch:
                    asset.setdefault("category", "other")
                    asset.setdefault("quality_score", 5)
                    asset.setdefault("is_usable", True)
                    classified.append(asset)

        return classified

    async def _classify_batch(
        self,
        batch: list[dict[str, Any]],
        website_url: str,
        brand_context: str,
    ) -> list[dict[str, Any]]:
        """Send a batch of images to Claude for classification."""
        content: list[dict[str, Any]] = []
        classifications: list[dict[str, Any]] = [
            self._fallback_classification(asset) for asset in batch
        ]
        analyzable_indices: list[int] = []

        # Build the message content with images + context
        for idx, asset in enumerate(batch):
            if asset.get("preclassified"):
                continue

            stored_path = asset.get("stored_url")
            if not stored_path:
                continue

            mime = asset.get("mime_type", "image/jpeg")
            if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
                continue

            image_data = await self._load_image_b64(
                stored_path,
                fallback_mime_type=asset.get("mime_type"),
            )
            if not image_data:
                continue

            analyzable_indices.append(idx)

            # Add context text for this image
            context_parts = []
            if asset.get("alt_text"):
                context_parts.append(f"Alt text: {asset['alt_text']}")
            if asset.get("context"):
                context_parts.append(f"Page context: {asset['context'][:300]}")
            if asset.get("source_page"):
                context_parts.append(f"Found on: {asset['source_page']}")

            content.append({
                "type": "text",
                "text": f"--- Image {idx} ---\n" + "\n".join(context_parts),
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": image_data,
                },
            })

        if not content:
            return classifications

        # Add the analysis prompt
        content.append({
            "type": "text",
            "text": BATCH_USER_PROMPT.format(
                count=len(batch),
                website_url=website_url,
                brand_context=brand_context[:500] if brand_context else "No additional context",
            ),
        })

        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        raw_text = response.content[0].text
        results = self._parse_results(raw_text, len(analyzable_indices))

        assigned_indices: set[int] = set()
        for result in results:
            result_index = result.get("index")
            if isinstance(result_index, int) and 0 <= result_index < len(batch):
                classifications[result_index].update(result)
                assigned_indices.add(result_index)

        if len(assigned_indices) != len(analyzable_indices):
            unassigned_results = [
                result
                for result in results
                if not isinstance(result.get("index"), int)
            ]
            for batch_index, result in zip(
                [idx for idx in analyzable_indices if idx not in assigned_indices],
                unassigned_results,
            ):
                classifications[batch_index].update(result)

        return classifications

    async def _load_image_b64(
        self,
        stored_url: str,
        fallback_mime_type: str | None = None,
    ) -> str | None:
        """Load an image from storage and return a base64 encoded string."""
        try:
            data, _ = await self._storage.read_asset(
                stored_url,
                fallback_content_type=fallback_mime_type,
            )
            # Skip files larger than 5MB to stay within API limits
            if len(data) > 5 * 1024 * 1024:
                return None
            return base64.standard_b64encode(data).decode("ascii")
        except Exception:
            logger.debug("Failed to load image %s", stored_url)
            return None

    @staticmethod
    def _parse_results(text: str, expected_count: int) -> list[dict[str, Any]]:
        """Parse the JSON array from Claude's response."""
        text = text.strip()

        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try fenced code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try finding array brackets
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                parsed = json.loads(bracket_match.group(0))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: return defaults
        logger.warning("Could not parse asset classification response")
        return [
            {"category": "other", "quality_score": 5, "is_usable": True}
        ] * expected_count

    @staticmethod
    def _fallback_classification(asset: dict[str, Any]) -> dict[str, Any]:
        """Return a safe default classification, preserving extractor hints."""
        if asset.get("preclassified"):
            return {
                "category": asset.get("category", "other"),
                "description": asset.get("description", ""),
                "quality_score": asset.get("quality_score", 5),
                "tags": asset.get("tags", []),
                "is_usable": asset.get("is_usable", False),
                "suggested_ad_use": asset.get("suggested_ad_use", ""),
            }

        mime_type = str(asset.get("mime_type") or "").lower()
        source = " ".join(
            [
                str(asset.get("source_url") or ""),
                str(asset.get("alt_text") or ""),
                str(asset.get("context") or ""),
            ]
        ).lower()
        category = "logo" if "logo" in source or "svg" in mime_type else "other"
        return {
            "category": category,
            "description": asset.get("description", ""),
            "quality_score": asset.get("quality_score", 5 if category == "logo" else 3),
            "tags": asset.get("tags", []),
            "is_usable": asset.get("is_usable", category == "logo"),
            "suggested_ad_use": asset.get(
                "suggested_ad_use",
                "Use as a brand identity element." if category == "logo" else "",
            ),
        }
