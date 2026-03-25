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
            response = await self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_text = response.content[0].text
            analysis = self._extract_json(raw_text)
            return analysis

        except Exception:
            logger.exception("LLM brand analysis failed for %s", website_url)
            raise
