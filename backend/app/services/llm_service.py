"""LLM service for brand analysis using Anthropic Claude."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert brand analyst. Given scraped website content, you extract \
structured brand intelligence that an advertising creative team can use to \
produce on-brand ad campaigns.

Always respond with a single JSON object — no commentary outside the JSON. \
If a field cannot be determined from the content, use null or an empty list \
rather than guessing."""

USER_PROMPT_TEMPLATE = """\
Analyze the following website content and any branding metadata. Return a \
JSON object with these exact keys:

{{
  "company_name": "string",
  "voice": {{
    "tone": "string — e.g. professional, playful, authoritative",
    "style": "string — e.g. concise, storytelling, data-driven",
    "personality_traits": ["list", "of", "traits"]
  }},
  "value_propositions": ["list of core value propositions"],
  "target_audience": {{
    "demographics": "string description",
    "pain_points": ["list"],
    "desires": ["list"]
  }},
  "products": [
    {{
      "name": "string",
      "description": "string",
      "key_benefits": ["list"]
    }}
  ],
  "industry": "string",
  "competitors": ["list of likely competitors"]
}}

---
Website URL: {website_url}

Branding metadata (scraped):
{branding_json}

Website content (markdown):
{markdown_content}
"""

CREATIVE_SYSTEM_PROMPT = """\
You are a performance creative strategist. You turn structured brand inputs \
into concrete ad concepts that a marketing team can immediately produce.

Always respond with a single JSON object and no surrounding commentary. \
Stay grounded in the supplied brand profile and listed assets. Do not invent \
social proof, product capabilities, or brand claims that are not supported \
by the input."""

CREATIVE_USER_PROMPT_TEMPLATE = """\
Create {concept_count} paid-media ad concepts for this brand. Return a JSON \
object with these exact keys:

{{
  "summary": "string",
  "brief": {{
    "primary_goal": "string",
    "audience_focus": "string",
    "offer_summary": "string",
    "messaging_pillars": ["list"],
    "tone_guardrails": ["list"],
    "visual_direction": ["list"],
    "recommended_formats": ["list"]
  }},
  "concepts": [
    {{
      "id": "concept-1",
      "name": "string",
      "format": "string",
      "angle": "string",
      "hook": "string",
      "primary_text": "string",
      "cta": "string",
      "why_it_will_work": "string",
      "visual_direction": ["list"],
      "asset_ids": ["use only IDs from the provided asset list"],
      "storyboard": [
        {{
          "step": "string",
          "detail": "string"
        }}
      ]
    }}
  ]
}}

Brand profile:
{brand_json}

Available assets:
{asset_json}
"""

EXECUTION_SYSTEM_PROMPT = """\
You are a senior performance creative director. You take one selected ad \
concept and expand it into a production-ready execution pack for copy, design, \
and video teams.

Always respond with a single JSON object and no surrounding commentary. Stay \
grounded in the provided brand brief and concept. Do not invent unsupported \
claims, customer numbers, guarantees, or product features."""

EXECUTION_USER_PROMPT_TEMPLATE = """\
Expand this concept into a production-ready execution pack. Return a JSON \
object with these exact keys:

{{
  "summary": "string",
  "headlines": ["3 to 5 headlines"],
  "primary_text_variants": ["2 to 4 body copy variants"],
  "ctas": ["3 to 5 CTA options"],
  "channel_variants": [
    {{
      "channel": "string",
      "format": "string",
      "headline": "string",
      "primary_text": "string",
      "cta": "string"
    }}
  ],
  "design_brief": {{
    "layout_direction": "string",
    "asset_strategy": "string",
    "copy_hierarchy": ["list"],
    "visual_notes": ["list"]
  }},
  "video_brief": {{
    "concept": "string",
    "opening_shot": "string",
    "shot_list": ["list"],
    "voiceover_script": "string",
    "end_frame": "string",
    "veo_prompt": "string"
  }},
  "production_checklist": ["list"]
}}

Brand profile:
{brand_json}

Creative brief:
{brief_json}

Selected concept:
{concept_json}

Available assets:
{asset_json}
"""


class LLMService:
    """Wraps the Anthropic SDK for brand-analysis tasks."""

    def __init__(self, anthropic_api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Parse JSON from a raw LLM response.

        Handles both a plain JSON body and JSON wrapped in a markdown
        code fence (```json ... ```).
        """
        # Try direct parse first.
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Look for a fenced code block.
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Last resort: find the first { … } blob.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not parse JSON from LLM response.")

    # ── public API ──────────────────────────────────────────────────────

    async def _request_json(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a prompt and parse the response as JSON."""
        response = await self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        return self._extract_json(raw_text)

    async def analyze_brand(
        self,
        website_url: str,
        markdown_content: str,
        branding_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Call Claude to produce a structured brand analysis.

        Parameters
        ----------
        website_url:
            The original URL (provided for context to the LLM).
        markdown_content:
            Scraped markdown text of the website.
        branding_data:
            Any branding hints already extracted (logo, colors, etc.).

        Returns
        -------
        dict
            Structured brand analysis matching the JSON schema above.
        """
        # Truncate very long content to stay within context limits.
        max_content_chars = 80_000
        if len(markdown_content) > max_content_chars:
            markdown_content = markdown_content[:max_content_chars] + "\n\n[content truncated]"

        user_message = USER_PROMPT_TEMPLATE.format(
            website_url=website_url,
            branding_json=json.dumps(branding_data, indent=2),
            markdown_content=markdown_content,
        )

        try:
            return await self._request_json(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
            )

        except Exception:
            logger.exception("LLM brand analysis failed for %s", website_url)
            raise

    async def generate_creative_studio(
        self,
        brand_context: dict[str, Any],
        asset_context: list[dict[str, Any]],
        concept_count: int,
    ) -> dict[str, Any]:
        """Generate a creative brief and concept set from a brand profile."""
        user_message = CREATIVE_USER_PROMPT_TEMPLATE.format(
            concept_count=concept_count,
            brand_json=json.dumps(brand_context, indent=2),
            asset_json=json.dumps(asset_context, indent=2),
        )

        try:
            return await self._request_json(
                system_prompt=CREATIVE_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=5000,
            )
        except Exception:
            logger.exception(
                "LLM creative studio generation failed for brand %s",
                brand_context.get("id", "unknown"),
            )
            raise

    async def generate_execution_pack(
        self,
        brand_context: dict[str, Any],
        brief: dict[str, Any],
        concept: dict[str, Any],
        asset_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate copy, design, and video outputs for one concept."""
        user_message = EXECUTION_USER_PROMPT_TEMPLATE.format(
            brand_json=json.dumps(brand_context, indent=2),
            brief_json=json.dumps(brief, indent=2),
            concept_json=json.dumps(concept, indent=2),
            asset_json=json.dumps(asset_context, indent=2),
        )

        try:
            return await self._request_json(
                system_prompt=EXECUTION_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=5000,
            )
        except Exception:
            logger.exception(
                "LLM execution pack generation failed for brand %s concept %s",
                brand_context.get("id", "unknown"),
                concept.get("id", "unknown"),
            )
            raise
