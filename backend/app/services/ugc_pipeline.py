"""UGC video pipeline — Flux.1 composite + OmniHuman talking head + Kling B-roll.

Completely isolated from the creative-studio / Veo pipeline.
Reuses GoogleTTSService, MubertMusicService, MediaComposerService, and AssetStorage.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from app.schemas.ugc import (
    UgcArtifact,
    UgcGenerateVideoRequest,
    UgcJobState,
    UgcPipelineStep,
    UgcScript,
    UgcScriptSegment,
    UgcVideoSettings,
)
from app.services.asset_storage import AssetStorage
from app.services.video_pipeline import (
    GoogleTTSService,
    MediaComposerService,
    MubertMusicService,
    RenderedBinary,
)

logger = logging.getLogger(__name__)


class UgcPipelineError(RuntimeError):
    """Raised when the UGC pipeline encounters an unrecoverable error."""


# ── fal.ai wrapper ──────────────────────────────────────────────────────


class FalAIService:
    """Async wrapper around fal.ai REST API for queue-based generation."""

    BASE_URL = "https://queue.fal.run"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key.strip()

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Key {self._api_key}",
            "Content-Type": "application/json",
        }

    async def submit_and_poll(
        self,
        endpoint: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: int = 600,
        poll_interval_seconds: int = 5,
    ) -> dict[str, Any]:
        """Submit a fal.ai queue job and poll until completion."""
        submit_url = f"{self.BASE_URL}/{endpoint}"
        logger.info("Submitting fal.ai job to %s", submit_url)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Submit
            resp = await client.post(
                submit_url,
                headers=self._headers(),
                json=arguments,
            )
            if resp.status_code >= 400:
                body = resp.text[:500]
                raise UgcPipelineError(
                    f"fal.ai submit failed ({resp.status_code}) for {endpoint}: {body}"
                )
            try:
                envelope = resp.json()
            except Exception:
                raise UgcPipelineError(
                    f"fal.ai submit returned non-JSON for {endpoint}: {resp.text[:500]}"
                )
            request_id = envelope.get("request_id")
            if not request_id:
                raise UgcPipelineError(f"fal.ai submit returned no request_id for {endpoint}")
            logger.info("fal.ai job submitted: %s (request_id=%s)", endpoint, request_id)

        # Poll using the response_url if provided, otherwise construct URLs
        status_url = envelope.get("status_url") or f"{self.BASE_URL}/{endpoint}/requests/{request_id}/status"
        response_url = envelope.get("response_url") or f"{self.BASE_URL}/{endpoint}/requests/{request_id}"

        elapsed = 0
        async with httpx.AsyncClient(timeout=30.0) as client:
            while elapsed < timeout_seconds:
                await asyncio.sleep(poll_interval_seconds)
                elapsed += poll_interval_seconds

                status_resp = await client.get(status_url, headers=self._headers())
                if status_resp.status_code >= 400:
                    logger.warning(
                        "fal.ai status poll failed (%s) for %s: %s",
                        status_resp.status_code, endpoint, status_resp.text[:200],
                    )
                    continue  # retry on transient errors
                try:
                    status_data = status_resp.json()
                except Exception:
                    logger.warning("fal.ai status returned non-JSON: %s", status_resp.text[:200])
                    continue
                poll_status = status_data.get("status", "")

                if poll_status == "COMPLETED":
                    result_resp = await client.get(response_url, headers=self._headers())
                    if result_resp.status_code >= 400:
                        body = result_resp.text[:500]
                        raise UgcPipelineError(
                            f"fal.ai result fetch failed ({result_resp.status_code}) for {endpoint}: {body}"
                        )
                    return result_resp.json()
                elif poll_status == "FAILED":
                    error = status_data.get("error", "Unknown fal.ai error")
                    raise UgcPipelineError(f"fal.ai job failed for {endpoint}: {error}")
                # IN_QUEUE or IN_PROGRESS — keep polling
                logger.info("fal.ai %s: %s (elapsed %ds)", endpoint, poll_status, elapsed)

        raise UgcPipelineError(f"fal.ai job timed out after {timeout_seconds}s for {endpoint}")

    async def _download_url(self, url: str) -> bytes:
        """Download a file from a URL."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def upload_file(self, data: bytes, content_type: str, file_name: str) -> str:
        """Upload a file to fal.ai CDN storage and return a public URL."""
        logger.info("Uploading %s (%d bytes, %s) to fal.ai CDN", file_name, len(data), content_type)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://fal.media/files/upload",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": content_type,
                    "X-Fal-File-Name": file_name,
                },
                content=data,
            )
            if resp.status_code >= 400:
                body = resp.text[:500]
                raise UgcPipelineError(
                    f"fal.ai upload failed ({resp.status_code}) for {file_name}: {body}"
                )
            try:
                result = resp.json()
            except Exception:
                raise UgcPipelineError(
                    f"fal.ai upload returned non-JSON for {file_name}: {resp.text[:300]}"
                )
            url = result.get("access_url", "") or result.get("url", "")
            if not url:
                raise UgcPipelineError(f"fal.ai upload returned no URL for {file_name}: {result}")
            logger.info("Uploaded %s → %s", file_name, url)
            return url

    # ── Flux.1 composite ────────────────────────────────────────────────

    async def generate_composite_image(
        self,
        *,
        avatar_image_url: str,
        product_image_url: str,
        prompt: str,
    ) -> bytes:
        """Flux.1 IP-Adapter: combine avatar portrait + product into composite reference."""
        result = await self.submit_and_poll(
            "fal-ai/flux-general/image-to-image",
            {
                "prompt": prompt,
                "image_url": avatar_image_url,
                "ip_adapters": [
                    {
                        "path": "https://huggingface.co/XLabs-AI/flux-ip-adapter-v2/resolve/main/ip_adapter.safetensors",
                        "image_encoder_path": "openai/clip-vit-large-patch14",
                        "image_url": product_image_url,
                        "scale": 0.4,
                    }
                ],
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
                "image_size": "portrait_16_9",
                "num_images": 1,
            },
            timeout_seconds=600,
        )
        images = result.get("images", [])
        if not images:
            raise UgcPipelineError("Flux.1 returned no images")
        return await self._download_url(images[0]["url"])

    # ── OmniHuman talking head ──────────────────────────────────────────

    async def generate_talking_head(
        self,
        *,
        reference_image_url: str,
        audio_url: str,
        resolution: str = "720p",
    ) -> bytes:
        """OmniHuman v1.5: animate composite reference with audio."""
        result = await self.submit_and_poll(
            "fal-ai/bytedance/omnihuman/v1.5",
            {
                "image_url": reference_image_url,
                "audio_url": audio_url,
                "resolution": resolution,
            },
            timeout_seconds=600,
            poll_interval_seconds=8,
        )
        video = result.get("video", {})
        video_url = video.get("url", "")
        if not video_url:
            raise UgcPipelineError("OmniHuman returned no video URL")
        return await self._download_url(video_url)

    # ── Kling B-roll ────────────────────────────────────────────────────

    async def generate_broll(
        self,
        *,
        product_image_url: str,
        prompt: str,
        duration_seconds: float = 3.0,
    ) -> bytes:
        """Kling v3 Pro: product-only B-roll shot."""
        result = await self.submit_and_poll(
            "fal-ai/kling-video/v3/pro/image-to-video",
            {
                "start_image_url": product_image_url,
                "prompt": prompt,
                "duration": str(min(int(duration_seconds), 15)),
                "generate_audio": False,
            },
            timeout_seconds=300,
            poll_interval_seconds=8,
        )
        video = result.get("video", {})
        video_url = video.get("url", "")
        if not video_url:
            raise UgcPipelineError("Kling returned no video URL")
        return await self._download_url(video_url)


# ── Script generation prompts ───────────────────────────────────────────

UGC_SCRIPT_SYSTEM_PROMPT = """\
You are an expert UGC video ad scriptwriter. You write short, punchy, \
conversational scripts for 20-30 second video ads where a person talks \
directly to camera while holding or using a product.

Always respond with a single JSON object — no commentary outside the JSON."""

UGC_SCRIPT_USER_PROMPT = """\
Write a UGC-style video ad script for {brand_name}'s product "{product_name}".

Product description: {product_description}
Target audience: {target_audience}
Key benefit to highlight: {key_benefit}
Desired tone: {tone}
CTA text: {cta_text}
Target duration: {target_duration_seconds} seconds

Return a JSON object:
{{
  "segments": [
    {{"role": "hook", "text": "...", "duration_hint_seconds": 5}},
    {{"role": "body", "text": "...", "duration_hint_seconds": 15}},
    {{"role": "cta", "text": "...", "duration_hint_seconds": 5}}
  ],
  "full_text": "Complete script as one paragraph for TTS",
  "estimated_duration_seconds": 25
}}

Rules:
- Hook must grab attention in the first 3 seconds
- Body should demonstrate the product benefit conversationally
- CTA must be clear and direct
- Total estimated duration should be close to {target_duration_seconds} seconds
- Write as if speaking to a friend, not reading an ad
"""


# ── Pipeline orchestrator ───────────────────────────────────────────────


class UgcPipelineService:
    """Orchestrates the full UGC video pipeline."""

    STEP_NAMES = ["script", "composite", "tts", "talking_head", "broll", "compose"]

    def __init__(
        self,
        *,
        fal: FalAIService,
        tts: GoogleTTSService,
        music: MubertMusicService,
        composer: MediaComposerService,
        storage: AssetStorage,
        llm: "LLMService | None" = None,
    ) -> None:
        self._fal = fal
        self._tts = tts
        self._music = music
        self._composer = composer
        self._storage = storage
        self._llm = llm

    # ── Step 1: Script ──────────────────────────────────────────────────

    async def generate_script(
        self,
        *,
        brand_name: str,
        product_name: str,
        product_description: str,
        target_audience: str,
        key_benefit: str,
        tone: str,
        cta_text: str,
        target_duration_seconds: int,
    ) -> tuple[UgcScript, bool]:
        """Generate a UGC script. Returns (script, used_fallback)."""
        if self._llm is not None:
            try:
                user_msg = UGC_SCRIPT_USER_PROMPT.format(
                    brand_name=brand_name,
                    product_name=product_name,
                    product_description=product_description or "N/A",
                    target_audience=target_audience or "general consumers",
                    key_benefit=key_benefit or "quality and value",
                    tone=tone or "conversational",
                    cta_text=cta_text or "Try it today",
                    target_duration_seconds=target_duration_seconds,
                )
                data = await self._llm._request_json(
                    system_prompt=UGC_SCRIPT_SYSTEM_PROMPT,
                    user_message=user_msg,
                    max_tokens=2048,
                )
                segments = [
                    UgcScriptSegment(
                        role=seg["role"],
                        text=seg["text"],
                        duration_hint_seconds=seg.get("duration_hint_seconds", 5),
                    )
                    for seg in data.get("segments", [])
                ]
                return UgcScript(
                    segments=segments,
                    full_text=data.get("full_text", " ".join(s.text for s in segments)),
                    estimated_duration_seconds=data.get(
                        "estimated_duration_seconds", target_duration_seconds
                    ),
                ), False
            except Exception as exc:
                logger.warning("LLM script generation failed, using fallback: %s", exc)

        return self._build_fallback_script(
            brand_name, product_name, key_benefit, cta_text, target_duration_seconds
        ), True

    @staticmethod
    def _build_fallback_script(
        brand_name: str,
        product_name: str,
        key_benefit: str,
        cta_text: str,
        target_duration_seconds: int,
    ) -> UgcScript:
        hook = f"Stop scrolling! You need to see this {product_name} from {brand_name}."
        body = (
            f"I've been using {product_name} for a week now and honestly, "
            f"the {key_benefit or 'quality'} is unreal. "
            f"If you've been looking for something that actually works, this is it."
        )
        cta = cta_text or f"Check out {brand_name} — link in bio!"
        segments = [
            UgcScriptSegment(role="hook", text=hook, duration_hint_seconds=5),
            UgcScriptSegment(
                role="body", text=body, duration_hint_seconds=max(target_duration_seconds - 10, 10)
            ),
            UgcScriptSegment(role="cta", text=cta, duration_hint_seconds=5),
        ]
        return UgcScript(
            segments=segments,
            full_text=f"{hook} {body} {cta}",
            estimated_duration_seconds=target_duration_seconds,
        )

    # ── Step 2: Composite reference image ───────────────────────────────

    async def generate_composite(
        self,
        *,
        avatar_image_url: str,
        product_image_url: str,
        brand_name: str,
        product_name: str,
        storage_prefix: str,
    ) -> UgcArtifact:
        """Create a reference image showing the avatar holding/using the product."""
        prompt = (
            f"A half-body portrait of a person from the waist up, wearing a casual t-shirt, "
            f"holding only a {product_name} in one hand near their chest, "
            f"looking directly at the camera and smiling naturally. "
            f"The person is NOT holding a phone, NOT wearing formal clothes or a suit. "
            f"Simple clean background, UGC selfie style filmed on phone, "
            f"natural indoor lighting, casual relaxed vibe, "
            f"face torso arms and hands clearly visible"
        )
        data = await self._fal.generate_composite_image(
            avatar_image_url=avatar_image_url,
            product_image_url=product_image_url,
            prompt=prompt,
        )
        file_name = "composite.png"
        key = f"{storage_prefix}/{file_name}"
        await self._storage.save_asset(key=key, data=data, content_type="image/png")
        return UgcArtifact(
            kind="composite_image",
            label="Avatar + Product Composite",
            stored_url=key,
            mime_type="image/png",
            file_name=file_name,
        )

    # ── Step 3: TTS voiceover ───────────────────────────────────────────

    async def generate_voiceover(
        self,
        *,
        script: UgcScript,
        settings: UgcVideoSettings,
        storage_prefix: str,
    ) -> UgcArtifact:
        """Generate TTS voiceover from the full script text."""
        if not self._tts.configured:
            raise UgcPipelineError("Google TTS is not configured.")

        audio_data = await self._tts.synthesize(
            script=script.full_text,
            voice_name=settings.tts_voice_name,
            speaking_rate=settings.tts_speaking_rate,
            pitch=settings.tts_pitch,
        )
        file_name = "voiceover.mp3"
        key = f"{storage_prefix}/{file_name}"
        await self._storage.save_asset(key=key, data=audio_data, content_type="audio/mpeg")
        return UgcArtifact(
            kind="voiceover",
            label="TTS Voiceover",
            stored_url=key,
            mime_type="audio/mpeg",
            file_name=file_name,
        )

    # ── Step 4: Talking head video ──────────────────────────────────────

    async def generate_talking_head(
        self,
        *,
        composite_stored_url: str,
        voiceover_stored_url: str,
        resolution: str,
        storage_prefix: str,
    ) -> UgcArtifact:
        """Animate the composite reference with the voiceover audio."""
        # fal.ai needs publicly accessible URLs — upload stored files to fal.ai storage
        composite_data, composite_ct = await self._storage.read_asset(composite_stored_url)
        voiceover_data, voiceover_ct = await self._storage.read_asset(voiceover_stored_url)

        composite_url, audio_url = await asyncio.gather(
            self._fal.upload_file(composite_data, composite_ct, "composite.png"),
            self._fal.upload_file(voiceover_data, voiceover_ct, "voiceover.mp3"),
        )

        video_data = await self._fal.generate_talking_head(
            reference_image_url=composite_url,
            audio_url=audio_url,
            resolution=resolution,
        )
        file_name = "talking_head.mp4"
        key = f"{storage_prefix}/{file_name}"
        await self._storage.save_asset(key=key, data=video_data, content_type="video/mp4")
        return UgcArtifact(
            kind="talking_head",
            label="Talking Head Video",
            stored_url=key,
            mime_type="video/mp4",
            file_name=file_name,
        )

    # ── Step 5: B-roll clips ────────────────────────────────────────────

    async def generate_broll_clips(
        self,
        *,
        product_image_url: str,
        product_name: str,
        brand_name: str,
        count: int,
        duration_seconds: float,
        storage_prefix: str,
    ) -> list[UgcArtifact]:
        """Generate product-focused B-roll clips in parallel."""
        broll_prompts = [
            f"Close-up product shot of {product_name}, smooth slow pan, studio lighting, premium feel",
            f"Lifestyle shot of {product_name} in use, natural warm lighting, modern aesthetic",
            f"Detail shot of {product_name} with shallow depth of field, rotating slowly",
        ]

        async def _gen_one(index: int) -> UgcArtifact:
            prompt = broll_prompts[index % len(broll_prompts)]
            data = await self._fal.generate_broll(
                product_image_url=product_image_url,
                prompt=prompt,
                duration_seconds=duration_seconds,
            )
            file_name = f"broll_{index:02d}.mp4"
            key = f"{storage_prefix}/{file_name}"
            await self._storage.save_asset(key=key, data=data, content_type="video/mp4")
            return UgcArtifact(
                kind="broll",
                label=f"B-roll {index + 1}",
                stored_url=key,
                mime_type="video/mp4",
                file_name=file_name,
            )

        tasks = [_gen_one(i) for i in range(count)]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # ── Step 6: Final composition ───────────────────────────────────────

    async def compose_final_video(
        self,
        *,
        talking_head_artifact: UgcArtifact,
        broll_artifacts: list[UgcArtifact],
        voiceover_artifact: UgcArtifact,
        script: UgcScript,
        settings: UgcVideoSettings,
        storage_prefix: str,
    ) -> UgcArtifact:
        """Stitch talking head + B-roll + music + captions into final video."""
        # Load talking head video
        th_data, th_ct = await self._storage.read_asset(talking_head_artifact.stored_url or "")
        th_binary = RenderedBinary(
            kind="talking_head", label="Talking Head", data=th_data,
            mime_type=th_ct, file_name="talking_head.mp4",
        )

        # Load B-roll clips
        broll_binaries: list[RenderedBinary] = []
        for br in broll_artifacts:
            br_data, br_ct = await self._storage.read_asset(br.stored_url or "")
            broll_binaries.append(RenderedBinary(
                kind="broll", label=br.label, data=br_data,
                mime_type=br_ct, file_name=br.file_name or "broll.mp4",
            ))

        # Concatenate: talking head with B-roll clips interleaved
        all_clips = [th_binary] + broll_binaries
        stitched_data = await self._composer.concatenate_videos(all_clips)
        stitched_binary = RenderedBinary(
            kind="stitched", label="Stitched", data=stitched_data,
            mime_type="video/mp4", file_name="stitched.mp4",
        )

        # Load voiceover
        vo_data, vo_ct = await self._storage.read_asset(voiceover_artifact.stored_url or "")
        vo_binary = RenderedBinary(
            kind="voiceover", label="Voiceover", data=vo_data,
            mime_type=vo_ct, file_name="voiceover.mp3",
        )

        # Generate music if enabled
        music_binary: RenderedBinary | None = None
        if settings.include_music and self._music.configured:
            try:
                music_data = await self._music.generate_track(
                    prompt=settings.music_prompt,
                    duration_seconds=settings.target_duration_seconds + 5,
                    intensity=settings.music_intensity,
                )
                music_binary = RenderedBinary(
                    kind="music", label="Background Music", data=music_data,
                    mime_type="audio/wav", file_name="music.wav",
                )
                music_key = f"{storage_prefix}/music.wav"
                await self._storage.save_asset(
                    key=music_key, data=music_data, content_type="audio/wav",
                )
            except Exception as exc:
                logger.warning("Music generation failed, proceeding without: %s", exc)

        # Compose final with audio
        final_data = await self._composer.compose_video(
            video=stitched_binary,
            voiceover=vo_binary,
            music=music_binary,
            include_native_audio=False,
            music_volume=0.20,
        )

        file_name = "final_ugc.mp4"
        key = f"{storage_prefix}/{file_name}"
        await self._storage.save_asset(key=key, data=final_data, content_type="video/mp4")
        return UgcArtifact(
            kind="final_video",
            label="Final UGC Video",
            stored_url=key,
            mime_type="video/mp4",
            file_name=file_name,
        )

    # ── Full pipeline orchestrator ──────────────────────────────────────

    async def run_pipeline(
        self,
        *,
        job: UgcJobState,
        request: UgcGenerateVideoRequest,
        brand_name: str,
        storage_prefix: str,
        on_update: Callable[[UgcJobState], None] | None = None,
    ) -> UgcJobState:
        """Run the full pipeline end-to-end, updating job state along the way."""

        def _step(name: str) -> UgcPipelineStep:
            return next(s for s in job.steps if s.step == name)

        def _mark(step_name: str, status: str, error: str | None = None) -> None:
            s = _step(step_name)
            s.status = status  # type: ignore[assignment]
            if status == "running":
                s.started_at = datetime.now(timezone.utc)
            elif status in ("completed", "failed"):
                s.completed_at = datetime.now(timezone.utc)
            s.error = error
            job.updated_at = datetime.now(timezone.utc)
            if on_update:
                on_update(job)

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        if on_update:
            on_update(job)

        # Ensure product and avatar images are publicly accessible for fal.ai
        product_image_url = await self._ensure_public_url(
            request.product_image_url, "product.jpg", "image/jpeg",
        )
        avatar_image_url = await self._ensure_public_url(
            request.avatar.image_url, "avatar.jpg", "image/jpeg",
        )

        try:
            # Script is already provided in request — mark done
            _mark("script", "completed")

            # Steps 2 + 3 in parallel: composite + TTS
            _mark("composite", "running")
            _mark("tts", "running")

            composite_task = self.generate_composite(
                avatar_image_url=avatar_image_url,
                product_image_url=product_image_url,
                brand_name=brand_name,
                product_name=request.product_name or "product",
                storage_prefix=storage_prefix,
            )
            voiceover_task = self.generate_voiceover(
                script=request.script,
                settings=request.settings,
                storage_prefix=storage_prefix,
            )
            composite_artifact, voiceover_artifact = await asyncio.gather(
                composite_task, voiceover_task
            )
            job.artifacts.extend([composite_artifact, voiceover_artifact])
            _mark("composite", "completed")
            _mark("tts", "completed")

            # Steps 4 + 5 in parallel: talking head + B-roll
            _mark("talking_head", "running")
            _mark("broll", "running")

            talking_head_task = self.generate_talking_head(
                composite_stored_url=composite_artifact.stored_url or "",
                voiceover_stored_url=voiceover_artifact.stored_url or "",
                resolution=request.settings.resolution,
                storage_prefix=storage_prefix,
            )
            broll_task = self.generate_broll_clips(
                product_image_url=product_image_url,
                product_name=request.product_name or "product",
                brand_name=brand_name,
                count=request.settings.broll_count,
                duration_seconds=request.settings.broll_duration_seconds,
                storage_prefix=storage_prefix,
            )
            talking_head_artifact, broll_artifacts = await asyncio.gather(
                talking_head_task, broll_task
            )
            job.artifacts.append(talking_head_artifact)
            job.artifacts.extend(broll_artifacts)
            _mark("talking_head", "completed")
            _mark("broll", "completed")

            # Step 6: compose
            _mark("compose", "running")
            final_artifact = await self.compose_final_video(
                talking_head_artifact=talking_head_artifact,
                broll_artifacts=broll_artifacts,
                voiceover_artifact=voiceover_artifact,
                script=request.script,
                settings=request.settings,
                storage_prefix=storage_prefix,
            )
            job.artifacts.append(final_artifact)
            job.final_video_url = final_artifact.stored_url
            _mark("compose", "completed")

            job.status = "completed"

        except Exception as exc:
            logger.exception("UGC pipeline failed: %s", exc)
            job.status = "failed"
            job.error = str(exc)
            # Mark any running steps as failed
            for s in job.steps:
                if s.status == "running":
                    s.status = "failed"
                    s.error = str(exc)
                    s.completed_at = datetime.now(timezone.utc)

        job.updated_at = datetime.now(timezone.utc)
        if on_update:
            on_update(job)
        return job

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _ensure_public_url(
        self, url_or_key: str, file_name: str, content_type: str,
    ) -> str:
        """Ensure a URL is publicly accessible by fal.ai.

        If it's already an HTTP URL, return as-is.
        If it's a local storage key, read the file and upload to fal.ai storage.
        """
        if not url_or_key:
            raise UgcPipelineError(
                f"No image URL provided for {file_name}. "
                "Upload a custom avatar portrait — built-in avatars need image files."
            )
        if url_or_key.startswith("http"):
            return url_or_key
        # It's a local storage key — read and upload to fal.ai
        data, ct = await self._storage.read_asset(url_or_key)
        return await self._fal.upload_file(data, ct or content_type, file_name)
