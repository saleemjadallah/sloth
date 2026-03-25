"""Builds a brand-specific creative brief and concept set."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.models.brand import Brand
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class CreativeStudioService:
    """Generates a creative brief and concepts for a brand."""

    _CATEGORY_PRIORITY = {
        "product": 5,
        "ui": 5,
        "screenshot": 4,
        "logo": 4,
        "lifestyle": 3,
        "team": 2,
    }

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm = llm_service

    async def build_studio(
        self,
        brand: Brand,
        concept_count: int = 4,
    ) -> dict[str, Any]:
        """Return creative strategy and concepts for the given brand."""
        asset_context = self._serialize_assets(brand)
        fallback_payload = self._build_fallback_payload(brand, asset_context, concept_count)

        payload = fallback_payload
        used_fallback = True

        if self._llm is not None:
            try:
                llm_payload = await self._llm.generate_creative_studio(
                    brand_context=self._serialize_brand(brand),
                    asset_context=asset_context,
                    concept_count=concept_count,
                )
                payload = self._normalize_payload(
                    llm_payload,
                    brand,
                    asset_context,
                    concept_count,
                    fallback_payload,
                )
                used_fallback = False
            except Exception:
                logger.exception("Creative studio generation failed for brand %s", brand.id)
                payload = fallback_payload

        payload["brand_id"] = brand.id
        payload["brand_name"] = brand.name or brand.website_url
        payload["generated_at"] = datetime.now(timezone.utc)
        payload["used_fallback"] = used_fallback
        return payload

    def _normalize_payload(
        self,
        payload: dict[str, Any],
        brand: Brand,
        asset_context: list[dict[str, Any]],
        concept_count: int,
        fallback_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Clean partial LLM output and guarantee a usable response."""
        valid_asset_ids = {asset["id"] for asset in asset_context}
        fallback_concepts = fallback_payload["concepts"]

        normalized = {
            "summary": str(payload.get("summary") or fallback_payload["summary"]),
            "brief": {
                "primary_goal": str(
                    payload.get("brief", {}).get("primary_goal")
                    or fallback_payload["brief"]["primary_goal"]
                ),
                "audience_focus": str(
                    payload.get("brief", {}).get("audience_focus")
                    or fallback_payload["brief"]["audience_focus"]
                ),
                "offer_summary": str(
                    payload.get("brief", {}).get("offer_summary")
                    or fallback_payload["brief"]["offer_summary"]
                ),
                "messaging_pillars": self._clean_list(
                    payload.get("brief", {}).get("messaging_pillars"),
                    fallback_payload["brief"]["messaging_pillars"],
                ),
                "tone_guardrails": self._clean_list(
                    payload.get("brief", {}).get("tone_guardrails"),
                    fallback_payload["brief"]["tone_guardrails"],
                ),
                "visual_direction": self._clean_list(
                    payload.get("brief", {}).get("visual_direction"),
                    fallback_payload["brief"]["visual_direction"],
                ),
                "recommended_formats": self._clean_list(
                    payload.get("brief", {}).get("recommended_formats"),
                    fallback_payload["brief"]["recommended_formats"],
                ),
            },
            "concepts": [],
        }

        raw_concepts = payload.get("concepts")
        if not isinstance(raw_concepts, list):
            raw_concepts = []

        for index, concept in enumerate(raw_concepts[:concept_count]):
            if not isinstance(concept, dict):
                continue
            fallback_concept = fallback_concepts[min(index, len(fallback_concepts) - 1)]
            raw_asset_ids = concept.get("asset_ids")
            asset_ids = [
                asset_id
                for asset_id in self._clean_list(raw_asset_ids)
                if asset_id in valid_asset_ids
            ] or fallback_concept["asset_ids"]

            storyboard = concept.get("storyboard")
            cleaned_storyboard = []
            if isinstance(storyboard, list):
                for beat in storyboard[:4]:
                    if not isinstance(beat, dict):
                        continue
                    step = str(beat.get("step") or "").strip()
                    detail = str(beat.get("detail") or "").strip()
                    if step and detail:
                        cleaned_storyboard.append({"step": step, "detail": detail})

            normalized["concepts"].append(
                {
                    "id": str(concept.get("id") or f"concept-{index + 1}"),
                    "name": str(concept.get("name") or fallback_concept["name"]),
                    "format": str(concept.get("format") or fallback_concept["format"]),
                    "angle": str(concept.get("angle") or fallback_concept["angle"]),
                    "hook": str(concept.get("hook") or fallback_concept["hook"]),
                    "primary_text": str(
                        concept.get("primary_text") or fallback_concept["primary_text"]
                    ),
                    "cta": str(concept.get("cta") or fallback_concept["cta"]),
                    "why_it_will_work": str(
                        concept.get("why_it_will_work")
                        or fallback_concept["why_it_will_work"]
                    ),
                    "visual_direction": self._clean_list(
                        concept.get("visual_direction"),
                        fallback_concept["visual_direction"],
                    ),
                    "asset_ids": asset_ids,
                    "storyboard": cleaned_storyboard or fallback_concept["storyboard"],
                }
            )

        if len(normalized["concepts"]) < concept_count:
            existing_ids = {concept["id"] for concept in normalized["concepts"]}
            for fallback_concept in fallback_concepts:
                if fallback_concept["id"] in existing_ids:
                    continue
                normalized["concepts"].append(fallback_concept)
                if len(normalized["concepts"]) == concept_count:
                    break

        if not normalized["summary"]:
            normalized["summary"] = (
                f"{brand.name or brand.website_url} is ready for concept development."
            )

        return normalized

    def _build_fallback_payload(
        self,
        brand: Brand,
        asset_context: list[dict[str, Any]],
        concept_count: int,
    ) -> dict[str, Any]:
        """Build a deterministic creative plan when LLM output is unavailable."""
        voice = brand.voice or {}
        audience = brand.target_audience or {}
        products = brand.products or []
        value_props = self._clean_list(brand.value_propositions)

        primary_product = products[0].get("name") if products else brand.name or "the brand"
        primary_benefit = (
            products[0].get("key_benefits", [None])[0]
            if products and products[0].get("key_benefits")
            else None
        )
        primary_value = primary_benefit or (value_props[0] if value_props else "clear practical value")
        audience_focus = (
            audience.get("demographics")
            or f"People evaluating {brand.industry or 'modern'} solutions"
        )
        pains = self._clean_list(audience.get("pain_points"))
        desires = self._clean_list(audience.get("desires"))

        top_assets = asset_context[:4]
        product_assets = [
            asset["id"]
            for asset in asset_context
            if asset.get("category") in {"product", "ui", "screenshot"}
        ]
        brand_assets = [asset["id"] for asset in top_assets]
        demo_assets = product_assets or brand_assets
        proof_problem = pains[0] if pains else "wasted time and inconsistent execution"
        outcome_desire = desires[0] if desires else "a faster path to confident results"

        concepts = [
            {
                "id": "concept-1",
                "name": "Proof In One Screen",
                "format": "static-image",
                "angle": f"Lead with the clearest benefit of {primary_product}.",
                "hook": f"Make {primary_value} obvious in the first glance.",
                "primary_text": (
                    f"{brand.name or primary_product} helps {audience_focus.lower()} move "
                    f"from {proof_problem} to {outcome_desire} without extra complexity."
                ),
                "cta": "See How It Works",
                "why_it_will_work": (
                    "This is the fastest path from brand profile to a high-signal prospecting ad."
                ),
                "visual_direction": [
                    "Keep the hero asset dominant and uncluttered.",
                    "Use one bold headline and one proof-oriented supporting line.",
                    "Lean on the extracted brand palette rather than introducing new colors.",
                ],
                "asset_ids": brand_assets[:2],
                "storyboard": [
                    {"step": "Hero", "detail": "Open with the strongest product or brand visual."},
                    {"step": "Proof", "detail": "Add one line that grounds the claim in a real benefit."},
                    {"step": "CTA", "detail": "Finish with a direct action-oriented button or footer."},
                ],
            },
            {
                "id": "concept-2",
                "name": "Pain To Outcome Carousel",
                "format": "carousel",
                "angle": "Turn the brand profile into a simple before/after narrative.",
                "hook": f"Start with the audience pain point: {proof_problem}.",
                "primary_text": (
                    f"Frame the audience challenge first, then show how {brand.name or primary_product} "
                    f"resolves it with a concrete, desirable outcome."
                ),
                "cta": "Explore The Workflow",
                "why_it_will_work": (
                    "Carousel structure creates room for narrative clarity without requiring new copy research."
                ),
                "visual_direction": [
                    "Slide 1 states the pain clearly.",
                    "Slides 2-3 show product, interface, or process visuals.",
                    "Final slide lands on the promised transformation and CTA.",
                ],
                "asset_ids": demo_assets[:3],
                "storyboard": [
                    {"step": "Pain", "detail": "Name the frustration in audience language."},
                    {"step": "Solution", "detail": "Show the product in action with a benefit callout."},
                    {"step": "Outcome", "detail": "End with the result the buyer wants."},
                ],
            },
            {
                "id": "concept-3",
                "name": "Founder Style UGC",
                "format": "short-video",
                "angle": "Explain the product like a smart person-to-camera recommendation.",
                "hook": f"\"If you need {outcome_desire.lower()}, stop doing it the hard way.\"",
                "primary_text": (
                    f"A short script that introduces the problem, positions {brand.name or primary_product} "
                    f"as the shortcut, and ends with one memorable proof point."
                ),
                "cta": "Try It For Yourself",
                "why_it_will_work": (
                    "UGC-style framing is the most flexible bridge into later storyboard and video production."
                ),
                "visual_direction": [
                    "Use candid framing, punchy captions, and fast first-three-second pacing.",
                    "Cut to supporting UI or product shots when the proof lands.",
                    "Keep the delivery aligned with the extracted brand voice.",
                ],
                "asset_ids": demo_assets[:2],
                "storyboard": [
                    {"step": "Hook", "detail": "Open with a clear, opinionated statement about the audience problem."},
                    {"step": "Demo", "detail": "Show the product or interface while naming the main benefit."},
                    {"step": "Close", "detail": "End with a direct CTA and one memorable proof line."},
                ],
            },
            {
                "id": "concept-4",
                "name": "Benefit Stack Demo",
                "format": "short-video",
                "angle": "Sequence the brand's best benefits into a quick motion-led walkthrough.",
                "hook": f"Three reasons {brand.name or primary_product} stands out.",
                "primary_text": (
                    f"Use a concise sequence of benefits drawn from the product set and value props, "
                    f"anchored on {primary_value}."
                ),
                "cta": "Book A Demo",
                "why_it_will_work": (
                    "This creates a practical storyboard for video production while staying grounded in available assets."
                ),
                "visual_direction": [
                    "Open on the cleanest UI or product shot available.",
                    "Use one distinct visual for each benefit beat.",
                    "End on a branded lockup and CTA frame.",
                ],
                "asset_ids": demo_assets[:3],
                "storyboard": [
                    {"step": "Intro", "detail": "Lead with brand and product recognition."},
                    {"step": "Benefits", "detail": "Move through three short proof-driven feature beats."},
                    {"step": "CTA", "detail": "Resolve with a branded end frame and next step."},
                ],
            },
        ]

        offer_summary = primary_value
        tone = voice.get("tone") or "confident"
        style = voice.get("style") or "clear"

        return {
            "summary": (
                f"{brand.name or brand.website_url} has enough brand signal to move from analysis "
                "into concept development and creative production."
            ),
            "brief": {
                "primary_goal": "Convert brand analysis into ready-to-produce ad concepts.",
                "audience_focus": audience_focus,
                "offer_summary": str(offer_summary),
                "messaging_pillars": value_props[:3] or [primary_value, outcome_desire, "clear differentiation"],
                "tone_guardrails": [
                    f"Stay {tone}.",
                    f"Keep the copy {style}.",
                    "Avoid generic hype that the current brand profile cannot support.",
                ],
                "visual_direction": [
                    "Prefer usable extracted assets before inventing new scenes.",
                    "Keep layouts clean enough for quick iteration across channels.",
                    "Use product, interface, or proof visuals wherever possible.",
                ],
                "recommended_formats": ["static-image", "carousel", "short-video"],
            },
            "concepts": concepts[:concept_count],
        }

    def _serialize_brand(self, brand: Brand) -> dict[str, Any]:
        """Return only the brand fields needed for creative generation."""
        return {
            "id": str(brand.id),
            "name": brand.name,
            "website_url": brand.website_url,
            "industry": brand.industry,
            "colors": brand.colors or {},
            "fonts": brand.fonts or {},
            "voice": brand.voice or {},
            "value_propositions": brand.value_propositions or [],
            "target_audience": brand.target_audience or {},
            "products": brand.products or [],
        }

    def _serialize_assets(self, brand: Brand) -> list[dict[str, Any]]:
        """Prepare the highest-signal asset set for prompting and UI mapping."""
        assets = list(brand.assets or [])
        assets.sort(
            key=lambda asset: (
                0 if asset.is_usable else 1,
                -self._CATEGORY_PRIORITY.get((asset.category or "").lower(), 0),
                -(asset.quality_score or 0),
                asset.file_name or "",
            )
        )

        serialized = []
        for asset in assets[:8]:
            serialized.append(
                {
                    "id": str(asset.id),
                    "category": asset.category or "",
                    "description": asset.description or "",
                    "is_usable": asset.is_usable,
                    "quality_score": asset.quality_score or 0,
                    "source_url": asset.source_url,
                    "stored_url": asset.stored_url,
                    "context": asset.context or "",
                }
            )
        return serialized

    @staticmethod
    def _clean_list(
        items: Any,
        fallback: list[str] | None = None,
    ) -> list[str]:
        """Normalize list-like values to trimmed strings."""
        if not isinstance(items, list):
            return list(fallback or [])
        normalized = [str(item).strip() for item in items if str(item).strip()]
        return normalized or list(fallback or [])
