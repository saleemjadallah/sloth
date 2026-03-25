"""Image variation generation backed by Google's Gemini image models."""

from __future__ import annotations

import asyncio
import base64
import io
import uuid
from pathlib import Path
from typing import Any

from PIL import Image


class ImageVariationService:
    """Generate prompted variations from an existing source asset."""

    DEFAULT_MODEL = "gemini-3-pro-image-preview"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed in the backend environment."
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model

    async def create_variation(
        self,
        *,
        source_bytes: bytes,
        source_mime_type: str,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Generate one or more images using the supplied source image + prompt."""
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=[
                self._build_prompt(prompt),
                self._types.Part.from_bytes(
                    data=source_bytes,
                    mime_type=source_mime_type,
                ),
            ],
        )

        results: list[dict[str, Any]] = []
        for part in self._iter_response_parts(response):
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue

            data = getattr(inline_data, "data", None)
            if not data:
                continue
            if isinstance(data, str):
                data = base64.b64decode(data)

            mime_type = getattr(inline_data, "mime_type", None) or "image/png"
            width, height = self._get_image_dimensions(data)
            results.append(
                {
                    "bytes": data,
                    "mime_type": mime_type,
                    "width": width,
                    "height": height,
                }
            )

        if not results:
            raise RuntimeError("Gemini returned no image output for the requested variation.")

        return results

    def save_generated_image(
        self,
        *,
        brand_id: str,
        image_bytes: bytes,
        mime_type: str,
        storage_root: str = "assets",
    ) -> dict[str, Any]:
        """Persist a generated image in the same storage layout as extracted assets."""
        extension = self._extension_for_mime_type(mime_type)
        brand_dir = Path(storage_root) / brand_id
        brand_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"generated-{uuid.uuid4().hex[:12]}{extension}"
        file_path = brand_dir / file_name
        file_path.write_bytes(image_bytes)

        width, height = self._get_image_dimensions(image_bytes)

        return {
            "stored_url": str(file_path),
            "file_name": file_name,
            "file_size": len(image_bytes),
            "mime_type": mime_type,
            "width": width,
            "height": height,
        }

    @staticmethod
    def _build_prompt(user_prompt: str) -> str:
        return (
            "Create a new ad-ready variation of the supplied source image. "
            "Preserve the core product, brand cues, and overall recognizability unless the prompt explicitly asks to change them. "
            "Return only the edited/generated image. "
            f"User direction: {user_prompt.strip()}"
        )

    @staticmethod
    def _iter_response_parts(response: Any) -> list[Any]:
        if getattr(response, "parts", None):
            return list(response.parts)

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []

        content = getattr(candidates[0], "content", None)
        return list(getattr(content, "parts", None) or [])

    @staticmethod
    def _get_image_dimensions(data: bytes) -> tuple[int | None, int | None]:
        try:
            with Image.open(io.BytesIO(data)) as image:
                return image.size
        except Exception:
            return None, None

    @staticmethod
    def _extension_for_mime_type(mime_type: str) -> str:
        return {
            "image/png": ".png",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime_type.lower(), ".png")
