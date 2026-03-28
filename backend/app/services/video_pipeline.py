"""Integrated video rendering pipeline for creative executions.

This service pulls together:
- Veo / Vertex video generation
- optional Google Cloud TTS voiceover
- optional Mubert background music
- ffmpeg-based stitching and audio composition
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import mimetypes
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from app.services.asset_storage import AssetStorage


@dataclass
class RenderedBinary:
    """In-memory binary produced by the pipeline."""

    kind: str
    label: str
    data: bytes
    mime_type: str
    file_name: str
    source_gcs_uri: str | None = None
    scene_id: str | None = None
    stored_url: str | None = None
    asset_id: str | None = None


@dataclass
class VideoPipelineResult:
    """Full pipeline result before persistence into the app DB/storage."""

    state: dict[str, Any]
    scene_videos: list[RenderedBinary]
    stitched_video: RenderedBinary | None
    voiceover: RenderedBinary | None
    music: RenderedBinary | None
    final_video: RenderedBinary | None


class VideoPipelineError(RuntimeError):
    """Raised when the pipeline cannot complete a requested render."""


class VeoVideoService:
    """Thin async wrapper around Vertex Veo predictLongRunning endpoints."""

    def __init__(
        self,
        *,
        project_id: str,
        access_token: str,
        gcs_bucket: str,
        location: str = "us-central1",
        default_model_id: str = "veo-3.1-generate-preview",
    ) -> None:
        self._project_id = project_id.strip()
        self._access_token = access_token.strip()
        self._gcs_bucket = gcs_bucket.strip().rstrip("/")
        self._location = location.strip() or "us-central1"
        self._default_model_id = default_model_id.strip() or "veo-3.1-generate-preview"

    @property
    def configured(self) -> bool:
        return bool(self._project_id and self._access_token and self._gcs_bucket)

    def _predict_url(self, model_id: str) -> str:
        return (
            "https://"
            f"{self._location}-aiplatform.googleapis.com/v1/projects/{self._project_id}"
            f"/locations/{self._location}/publishers/google/models/{model_id}:predictLongRunning"
        )

    def _fetch_operation_url(self, model_id: str) -> str:
        return (
            "https://"
            f"{self._location}-aiplatform.googleapis.com/v1/projects/{self._project_id}"
            f"/locations/{self._location}/publishers/google/models/{model_id}:fetchPredictOperation"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": self._project_id,
        }

    def storage_uri(self, storage_prefix: str) -> str:
        base = self._gcs_bucket
        if not base.startswith("gs://"):
            base = f"gs://{base.lstrip('/')}"
        return f"{base.rstrip('/')}/{storage_prefix.strip('/').rstrip('/')}/"

    async def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=self._headers(), json=body)
            response.raise_for_status()
            return response.json()

    async def _submit_generation(
        self,
        *,
        instances: list[dict[str, Any]],
        parameters: dict[str, Any],
        model_id: str,
    ) -> dict[str, Any]:
        return await self._post(
            self._predict_url(model_id),
            {"instances": instances, "parameters": parameters},
        )

    async def poll_operation(
        self,
        *,
        operation_name: str,
        model_id: str,
        timeout_seconds: int = 1200,
        poll_interval_seconds: int = 5,
    ) -> dict[str, Any]:
        attempts = max(1, timeout_seconds // poll_interval_seconds)
        for _ in range(attempts):
            await asyncio.sleep(poll_interval_seconds)
            result = await self._post(
                self._fetch_operation_url(model_id),
                {"operationName": operation_name},
            )
            if result.get("done"):
                return result
        raise VideoPipelineError("Timed out waiting for Veo render operation to complete.")

    @staticmethod
    def _walk_media_nodes(node: Any) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        if isinstance(node, dict):
            if node.get("gcsUri") or node.get("bytesBase64Encoded"):
                matches.append(node)
            for value in node.values():
                matches.extend(VeoVideoService._walk_media_nodes(value))
        elif isinstance(node, list):
            for item in node:
                matches.extend(VeoVideoService._walk_media_nodes(item))
        return matches

    @classmethod
    def extract_video_payload(cls, result: dict[str, Any]) -> dict[str, Any]:
        for candidate in cls._walk_media_nodes(result):
            if candidate.get("gcsUri"):
                return {
                    "gcs_uri": str(candidate["gcsUri"]),
                    "bytes": None,
                    "mime_type": str(candidate.get("mimeType") or "video/mp4"),
                }
            if candidate.get("bytesBase64Encoded"):
                return {
                    "gcs_uri": None,
                    "bytes": base64.b64decode(str(candidate["bytesBase64Encoded"])),
                    "mime_type": str(candidate.get("mimeType") or "video/mp4"),
                }
        if result.get("response", {}).get("raiMediaFilteredCount", 0) > 0:
            reasons = result.get("response", {}).get("raiMediaFilteredReasons") or []
            joined = ", ".join(reasons) if isinstance(reasons, list) else "unknown"
            raise VideoPipelineError(f"Veo output was filtered by safety policies: {joined}")
        raise VideoPipelineError("Veo completed without returning video media in the operation payload.")

    async def generate(
        self,
        *,
        prompt: str,
        model_id: str,
        parameters: dict[str, Any],
        image: dict[str, str] | None = None,
        last_frame: dict[str, str] | None = None,
        reference_images: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        instance: dict[str, Any] = {"prompt": prompt}
        if image is not None:
            instance["image"] = image
        if last_frame is not None:
            instance["lastFrame"] = last_frame
        if reference_images:
            instance["referenceImages"] = reference_images
        return await self._submit_generation(
            instances=[instance],
            parameters=parameters,
            model_id=model_id,
        )

    async def extend(
        self,
        *,
        prompt: str,
        video_gcs_uri: str,
        model_id: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._submit_generation(
            instances=[
                {
                    "prompt": prompt,
                    "video": {"gcsUri": video_gcs_uri, "mimeType": "video/mp4"},
                }
            ],
            parameters=parameters,
            model_id=model_id,
        )

    async def download_gcs_uri(self, gcs_uri: str) -> bytes:
        if not gcs_uri.startswith("gs://"):
            raise VideoPipelineError(f"Unsupported GCS URI: {gcs_uri}")
        without_prefix = gcs_uri.removeprefix("gs://")
        bucket, _, object_name = without_prefix.partition("/")
        if not bucket or not object_name:
            raise VideoPipelineError(f"Malformed GCS URI: {gcs_uri}")
        encoded_name = quote(object_name, safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{encoded_name}?alt=media"
        async with httpx.AsyncClient(timeout=240.0) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.content


class GoogleTTSService:
    """Google Cloud TTS helper backed by service-account credentials."""

    SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

    def __init__(
        self,
        *,
        credentials_json: str = "",
        credentials_path: str | None = None,
        default_voice_name: str = "en-US-Studio-O",
        default_pitch: float = 0.0,
        default_effects_profile_id: str = "headphone-class-device",
        max_script_chars: int = 50000,
    ) -> None:
        self._credentials_json = credentials_json.strip()
        self._credentials_path = (credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        self._default_voice_name = default_voice_name.strip() or "en-US-Studio-O"
        self._default_pitch = float(default_pitch)
        self._default_effects_profile_id = (
            default_effects_profile_id.strip() or "headphone-class-device"
        )
        self._max_script_chars = max(1000, int(max_script_chars or 50000))

    @property
    def configured(self) -> bool:
        return bool(self._credentials_json or self._credentials_path)

    @property
    def default_voice_name(self) -> str:
        return self._default_voice_name

    def _load_credentials(self) -> service_account.Credentials:
        if self._credentials_json:
            payload = json.loads(self._credentials_json)
            return service_account.Credentials.from_service_account_info(
                payload,
                scopes=self.SCOPES,
            )
        if self._credentials_path:
            return service_account.Credentials.from_service_account_file(
                self._credentials_path,
                scopes=self.SCOPES,
            )
        raise VideoPipelineError(
            "Google TTS is not configured. Set GOOGLE_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS."
        )

    async def _resolve_access_token(self) -> tuple[str, str | None]:
        credentials = await asyncio.to_thread(self._load_credentials)

        def _refresh() -> tuple[str, str | None]:
            scoped = credentials.with_scopes(self.SCOPES) if credentials.requires_scopes else credentials
            scoped.refresh(GoogleAuthRequest())
            return str(scoped.token or ""), scoped.project_id

        token, project_id = await asyncio.to_thread(_refresh)
        if not token:
            raise VideoPipelineError("Google TTS credentials could not mint an access token.")
        return token, project_id

    @staticmethod
    def _language_code_for_voice(voice_name: str) -> str:
        parts = [part for part in voice_name.split("-") if part]
        if len(parts) >= 2:
            return "-".join(parts[:2])
        return "en-US"

    async def synthesize(
        self,
        *,
        script: str,
        voice_name: str,
        speaking_rate: float,
        pitch: float | None = None,
        effects_profile_id: str | None = None,
    ) -> bytes:
        normalized_script = script.strip()
        if not normalized_script:
            raise VideoPipelineError("Voiceover was requested but the script is empty.")
        if len(normalized_script) > self._max_script_chars:
            raise VideoPipelineError(
                f"Voiceover script exceeds the {self._max_script_chars:,}-character safety limit."
            )

        access_token, project_id = await self._resolve_access_token()
        resolved_voice_name = voice_name.strip() or self._default_voice_name
        resolved_speaking_rate = min(max(float(speaking_rate or 1.0), 0.5), 2.0)
        resolved_pitch = min(max(float(self._default_pitch if pitch is None else pitch), -20.0), 20.0)
        resolved_effects_profile_id = (
            effects_profile_id.strip()
            if isinstance(effects_profile_id, str) and effects_profile_id.strip()
            else self._default_effects_profile_id
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        if project_id:
            headers["X-Goog-User-Project"] = project_id

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://texttospeech.googleapis.com/v1/text:synthesize",
                headers=headers,
                json={
                    "input": {"text": normalized_script},
                    "voice": {
                        "languageCode": self._language_code_for_voice(resolved_voice_name),
                        "name": resolved_voice_name,
                    },
                    "audioConfig": {
                        "audioEncoding": "MP3",
                        "speakingRate": resolved_speaking_rate,
                        "pitch": resolved_pitch,
                        "effectsProfileId": [resolved_effects_profile_id],
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
            audio_content = payload.get("audioContent")
            if not audio_content:
                raise VideoPipelineError("Google TTS returned no audio content.")
            return base64.b64decode(audio_content)


class MubertMusicService:
    """Simple Mubert track generation helper."""

    BASE_URL = "https://music-api.mubert.com/api/v3"

    def __init__(self, *, company_id: str, license_token: str) -> None:
        self._company_id = company_id.strip()
        self._license_token = license_token.strip()
        self._customer_id: str | None = None
        self._access_token: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self._company_id and self._license_token)

    async def _ensure_customer(self) -> tuple[str, str]:
        if self._customer_id and self._access_token:
            return self._customer_id, self._access_token

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/service/customers",
                headers={
                    "Content-Type": "application/json",
                    "company-id": self._company_id,
                    "license-token": self._license_token,
                },
                json={"custom_id": f"sloth-{uuid.uuid4().hex[:12]}"},
            )
            response.raise_for_status()
            payload = response.json()
            access = payload.get("data", {}).get("access", {})
            customer_id = str(access.get("customer_id") or "")
            token = str(access.get("token") or "")
            if not customer_id or not token:
                raise VideoPipelineError("Mubert customer session could not be created.")
            self._customer_id = customer_id
            self._access_token = token
            return customer_id, token

    async def generate_track(
        self,
        *,
        prompt: str,
        duration_seconds: int,
        intensity: str = "medium",
        mode: str = "track",
        fmt: str = "wav",
        bitrate: int = 320,
    ) -> bytes:
        customer_id, access_token = await self._ensure_customer()
        headers = {
            "Content-Type": "application/json",
            "customer-id": customer_id,
            "access-token": access_token,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            create_response = await client.post(
                f"{self.BASE_URL}/public/tracks",
                headers=headers,
                json={
                    "prompt": prompt[:200],
                    "duration": duration_seconds,
                    "bitrate": bitrate,
                    "format": fmt,
                    "intensity": intensity,
                    "mode": mode,
                },
            )
            create_response.raise_for_status()
            payload = create_response.json()
            track_id = payload.get("data", {}).get("id")
            generation = payload.get("data", {}).get("generations", [{}])[0]
            if generation.get("status") == "done" and generation.get("url"):
                download_url = generation["url"]
            else:
                download_url = await self._poll_track(track_id=track_id, headers=headers)

            if not download_url:
                raise VideoPipelineError("Mubert completed without a downloadable track URL.")

            audio_response = await client.get(download_url, timeout=240.0)
            audio_response.raise_for_status()
            return audio_response.content

    async def _poll_track(
        self,
        *,
        track_id: str,
        headers: dict[str, str],
        max_attempts: int = 60,
        interval_seconds: int = 3,
    ) -> str:
        if not track_id:
            raise VideoPipelineError("Mubert returned no track id.")
        async with httpx.AsyncClient(timeout=60.0) as client:
            for _ in range(max_attempts):
                await asyncio.sleep(interval_seconds)
                response = await client.get(
                    f"{self.BASE_URL}/public/tracks/{track_id}",
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
                generation = payload.get("data", {}).get("generations", [{}])[0]
                status = generation.get("status")
                if status == "done" and generation.get("url"):
                    return str(generation["url"])
                if status in {"failed", "error"}:
                    raise VideoPipelineError("Mubert music generation failed.")
        raise VideoPipelineError("Mubert music generation timed out.")


class MediaComposerService:
    """ffmpeg-based media composition helpers."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe") -> None:
        self._ffmpeg_path = ffmpeg_path
        self._ffprobe_path = ffprobe_path

    async def concatenate_videos(self, videos: list[RenderedBinary]) -> bytes:
        if not videos:
            raise VideoPipelineError("No videos were provided for concatenation.")
        if len(videos) == 1:
            return videos[0].data

        with tempfile.TemporaryDirectory(prefix="sloth-video-concat-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "concat.txt"
            lines: list[str] = []
            for index, artifact in enumerate(videos):
                clip_path = tmp_path / f"clip-{index:02d}.mp4"
                clip_path.write_bytes(artifact.data)
                lines.append(f"file '{clip_path.as_posix()}'")
            manifest_path.write_text("\n".join(lines), encoding="utf-8")
            output_path = tmp_path / "stitched.mp4"

            await self._run(
                [
                    self._ffmpeg_path,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(manifest_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )
            return output_path.read_bytes()

    async def compose_video(
        self,
        *,
        video: RenderedBinary,
        voiceover: RenderedBinary | None,
        music: RenderedBinary | None,
        include_native_audio: bool,
        native_audio_volume: float = 0.2,
        music_volume: float = 0.28,
    ) -> bytes:
        if voiceover is None and music is None and not include_native_audio:
            return video.data

        with tempfile.TemporaryDirectory(prefix="sloth-video-compose-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = tmp_path / "video.mp4"
            video_path.write_bytes(video.data)
            output_path = tmp_path / "final.mp4"

            command = [
                self._ffmpeg_path,
                "-y",
                "-i",
                str(video_path),
            ]
            filter_parts: list[str] = []
            mix_labels: list[str] = []
            input_index = 1

            if include_native_audio and await self._has_audio_stream(video_path):
                filter_parts.append(f"[0:a]volume={native_audio_volume}[a0]")
                mix_labels.append("[a0]")

            if voiceover is not None:
                voice_path = tmp_path / "voiceover.mp3"
                voice_path.write_bytes(voiceover.data)
                command.extend(["-i", str(voice_path)])
                filter_parts.append(f"[{input_index}:a]volume=1.0[a{input_index}]")
                mix_labels.append(f"[a{input_index}]")
                input_index += 1

            if music is not None:
                music_ext = mimetypes.guess_extension(music.mime_type or "audio/wav") or ".wav"
                music_path = tmp_path / f"music{music_ext}"
                music_path.write_bytes(music.data)
                command.extend(["-i", str(music_path)])
                filter_parts.append(f"[{input_index}:a]volume={music_volume}[a{input_index}]")
                mix_labels.append(f"[a{input_index}]")
                input_index += 1

            if not mix_labels:
                return video.data

            if len(mix_labels) == 1:
                output_label = mix_labels[0]
            else:
                output_label = "[mixout]"
                filter_parts.append(
                    "".join(mix_labels)
                    + f"amix=inputs={len(mix_labels)}:duration=longest:normalize=0{output_label}"
                )

            command.extend(
                [
                    "-filter_complex",
                    ";".join(filter_parts),
                    "-map",
                    "0:v:0",
                    "-map",
                    output_label,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )

            await self._run(command)
            return output_path.read_bytes()

    async def _has_audio_stream(self, media_path: Path) -> bool:
        process = await asyncio.create_subprocess_exec(
            self._ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(media_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        return process.returncode == 0 and bool(stdout.strip())

    async def _run(self, command: list[str]) -> None:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="ignore").strip()
            raise VideoPipelineError(f"ffmpeg failed: {detail or 'unknown error'}")


class VideoPipelineService:
    """High-level orchestration for in-app video rendering."""

    DEFAULT_SETTINGS: dict[str, Any] = {
        "render_strategy": "scene_sequence",
        "generation_mode": "auto",
        "model_id": "veo-3.1-generate-preview",
        "aspect_ratio": "9:16",
        "resolution": "1080p",
        "scene_duration_seconds": 8,
        "negative_prompt": "",
        "seed": None,
        "generate_native_audio": False,
        "create_voiceover": True,
        "create_music": False,
        "compose_final": True,
        "stitch_scenes": True,
        "tts_voice_name": "en-US-Studio-O",
        "tts_speaking_rate": 1.0,
        "tts_pitch": 0.0,
        "tts_effects_profile_id": "headphone-class-device",
        "music_prompt": "",
        "music_intensity": "medium",
        "music_mode": "track",
        "music_format": "wav",
        "music_bitrate": 320,
        "reference_asset_ids": [],
        "first_frame_asset_id": None,
        "last_frame_asset_id": None,
    }

    def __init__(
        self,
        *,
        veo: VeoVideoService,
        tts: GoogleTTSService,
        music: MubertMusicService,
        composer: MediaComposerService,
        storage: AssetStorage,
    ) -> None:
        self._veo = veo
        self._tts = tts
        self._music = music
        self._composer = composer
        self._storage = storage

    @property
    def configured(self) -> bool:
        return self._veo.configured

    async def render_execution(
        self,
        *,
        brand_name: str,
        execution: dict[str, Any],
        selected_assets: list[dict[str, Any]],
        storage_prefix: str,
        regenerate_scenes: bool = False,
    ) -> VideoPipelineResult:
        if not self._veo.configured:
            raise VideoPipelineError(
                "Veo is not configured. Set VEO_PROJECT_ID, VEO_ACCESS_TOKEN, and VEO_GCS_BUCKET."
            )

        prior_state = execution.get("video_render")
        settings = {
            **self.DEFAULT_SETTINGS,
            **(prior_state.get("settings", {}) if isinstance(prior_state, dict) else {}),
        }
        model_id = str(settings.get("model_id") or self.DEFAULT_SETTINGS["model_id"])

        logs: list[str] = []
        scenes = (
            []
            if regenerate_scenes
            else list(prior_state.get("scenes", []) if isinstance(prior_state, dict) else [])
        )
        if not scenes:
            scenes = self.build_scene_plan(execution=execution, selected_assets=selected_assets, settings=settings)
            logs.append(f"Generated {len(scenes)} scene(s) from the current video brief.")

        enabled_scenes = [scene for scene in scenes if scene.get("enabled", True)]
        if not enabled_scenes:
            raise VideoPipelineError("At least one enabled scene is required to render video.")

        scene_videos: list[RenderedBinary] = []
        current_video_gcs_uri: str | None = None

        if settings.get("render_strategy") == "daisy_chain":
            for index, scene in enumerate(enabled_scenes):
                operation = await self._render_scene_or_extension(
                    scene=scene,
                    scene_index=index,
                    model_id=model_id,
                    settings=settings,
                    selected_assets=selected_assets,
                    storage_prefix=storage_prefix,
                    previous_video_gcs_uri=current_video_gcs_uri,
                )
                current_video_gcs_uri = operation["video_gcs_uri"]
                scene_videos.append(operation["artifact"])
                logs.extend(operation["logs"])
        else:
            for index, scene in enumerate(enabled_scenes):
                operation = await self._render_scene_or_extension(
                    scene=scene,
                    scene_index=index,
                    model_id=model_id,
                    settings=settings,
                    selected_assets=selected_assets,
                    storage_prefix=storage_prefix,
                    previous_video_gcs_uri=None,
                )
                scene_videos.append(operation["artifact"])
                logs.extend(operation["logs"])

        stitched_artifact: RenderedBinary | None = None
        if len(scene_videos) > 1 and settings.get("stitch_scenes", True):
            stitched_bytes = await self._composer.concatenate_videos(scene_videos)
            stitched_artifact = RenderedBinary(
                kind="stitched_video",
                label="Stitched scenes",
                data=stitched_bytes,
                mime_type="video/mp4",
                file_name="stitched-video.mp4",
            )
            logs.append("Stitched rendered scene clips into a single video.")

        base_video = stitched_artifact or scene_videos[-1]

        voiceover_artifact: RenderedBinary | None = None
        if settings.get("create_voiceover"):
            if not self._tts.configured:
                raise VideoPipelineError(
                    "Voiceover was requested but Google TTS is not configured with backend credentials."
                )
            voiceover_script = str(execution.get("video_brief", {}).get("voiceover_script") or "").strip()
            estimated_voiceover_seconds = self.estimate_voiceover_duration_seconds(
                script=voiceover_script,
                speaking_rate=float(settings.get("tts_speaking_rate") or 1.0),
            )
            voiceover_bytes = await self._tts.synthesize(
                script=voiceover_script,
                voice_name=str(settings.get("tts_voice_name") or self._tts.default_voice_name),
                speaking_rate=float(settings.get("tts_speaking_rate") or 1.0),
                pitch=float(settings.get("tts_pitch") or 0.0),
                effects_profile_id=str(settings.get("tts_effects_profile_id") or ""),
            )
            voiceover_artifact = RenderedBinary(
                kind="voiceover",
                label="Voiceover narration",
                data=voiceover_bytes,
                mime_type="audio/mpeg",
                file_name="voiceover.mp3",
            )
            logs.append(
                f"Generated voiceover narration from the execution script (~{estimated_voiceover_seconds}s at the selected speaking rate)."
            )

        music_artifact: RenderedBinary | None = None
        if settings.get("create_music"):
            if not self._music.configured:
                raise VideoPipelineError(
                    "Music generation was requested but MUBERT_COMPANY_ID / MUBERT_LICENSE_TOKEN are missing."
                )
            music_prompt = str(settings.get("music_prompt") or "").strip() or self._default_music_prompt(
                brand_name=brand_name,
                execution=execution,
            )
            voiceover_duration_seconds = self.estimate_voiceover_duration_seconds(
                script=str(execution.get("video_brief", {}).get("voiceover_script") or "").strip(),
                speaking_rate=float(settings.get("tts_speaking_rate") or 1.0),
            )
            duration_seconds = max(
                voiceover_duration_seconds,
                int(settings.get("scene_duration_seconds") or 8) * max(1, len(enabled_scenes)),
                8,
            )
            music_bytes = await self._music.generate_track(
                prompt=music_prompt,
                duration_seconds=duration_seconds,
                intensity=str(settings.get("music_intensity") or "medium"),
                mode=str(settings.get("music_mode") or "track"),
                fmt=str(settings.get("music_format") or "wav"),
                bitrate=int(settings.get("music_bitrate") or 320),
            )
            music_artifact = RenderedBinary(
                kind="music",
                label="Background music",
                data=music_bytes,
                mime_type="audio/wav" if str(settings.get("music_format")) == "wav" else "audio/mpeg",
                file_name=f"background-music.{ 'wav' if str(settings.get('music_format')) == 'wav' else 'mp3' }",
            )
            logs.append("Generated background music for the rendered video.")

        final_artifact: RenderedBinary | None = None
        if settings.get("compose_final") and (
            voiceover_artifact is not None or music_artifact is not None or settings.get("generate_native_audio")
        ):
            final_bytes = await self._composer.compose_video(
                video=base_video,
                voiceover=voiceover_artifact,
                music=music_artifact,
                include_native_audio=bool(settings.get("generate_native_audio")),
            )
            final_artifact = RenderedBinary(
                kind="final_video",
                label="Final composed video",
                data=final_bytes,
                mime_type="video/mp4",
                file_name="final-composed-video.mp4",
            )
            logs.append("Composed the final video with the requested audio layers.")
        else:
            final_artifact = base_video

        state = {
            "status": "completed",
            "provider": "vertex_veo",
            "settings": settings,
            "scenes": scenes,
            "logs": logs,
            "errors": [],
        }

        return VideoPipelineResult(
            state=state,
            scene_videos=scene_videos,
            stitched_video=stitched_artifact,
            voiceover=voiceover_artifact,
            music=music_artifact,
            final_video=final_artifact,
        )

    @staticmethod
    def estimate_voiceover_duration_seconds(*, script: str, speaking_rate: float) -> int:
        words = len([part for part in script.split() if part.strip()])
        if words == 0:
            return 0
        normalized_rate = min(max(float(speaking_rate or 1.0), 0.5), 2.0)
        return max(1, int(math.ceil((words / 150.0) * 60.0 / normalized_rate)))

    def build_scene_plan(
        self,
        *,
        execution: dict[str, Any],
        selected_assets: list[dict[str, Any]],
        settings: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        settings = settings or self.DEFAULT_SETTINGS
        video_brief = execution.get("video_brief", {}) if isinstance(execution.get("video_brief"), dict) else {}
        shot_list = [
            str(item).strip()
            for item in (video_brief.get("shot_list") or [])
            if str(item).strip()
        ]
        if not shot_list:
            shot_list = [
                str(video_brief.get("opening_shot") or "").strip(),
                str(video_brief.get("end_frame") or "").strip(),
            ]
            shot_list = [item for item in shot_list if item]

        base_prompt = str(video_brief.get("veo_prompt") or execution.get("summary") or "").strip()
        voiceover = str(video_brief.get("voiceover_script") or "").strip()
        concept_name = str(execution.get("concept_name") or video_brief.get("concept") or "Creative video")
        duration_seconds = int(settings.get("scene_duration_seconds") or 8)

        reference_asset_ids = [
            str(asset_id)
            for asset_id in (settings.get("reference_asset_ids") or [])
            if str(asset_id).strip()
        ]
        if not reference_asset_ids:
            reference_asset_ids = [
                str(asset.get("id"))
                for asset in selected_assets
                if str(asset.get("mime_type") or "").startswith("image/")
            ][:3]

        scenes: list[dict[str, Any]] = []
        for index, beat in enumerate(shot_list, start=1):
            scene_prompt = self._build_scene_prompt(
                master_prompt=base_prompt,
                shot=beat,
                scene_index=index,
                total_scenes=len(shot_list),
                concept_name=concept_name,
                opening_shot=str(video_brief.get("opening_shot") or "").strip(),
                end_frame=str(video_brief.get("end_frame") or "").strip(),
            )
            scenes.append(
                {
                    "id": f"scene-{index}",
                    "title": beat[:80] or f"Scene {index}",
                    "prompt": scene_prompt,
                    "duration_seconds": duration_seconds,
                    "voiceover_text": voiceover,
                    "reference_asset_ids": reference_asset_ids,
                    "enabled": True,
                    "sequence_index": index - 1,
                    "concept_name": concept_name,
                }
            )
        return scenes or [
            {
                "id": "scene-1",
                "title": concept_name,
                "prompt": base_prompt or concept_name,
                "duration_seconds": duration_seconds,
                "voiceover_text": voiceover,
                "reference_asset_ids": reference_asset_ids,
                "enabled": True,
                "sequence_index": 0,
                "concept_name": concept_name,
            }
        ]

    @staticmethod
    def _split_prompt_sentences(prompt: str) -> list[str]:
        normalized = prompt.replace("\n", " ").strip()
        if not normalized:
            return []
        return [part.strip() for part in normalized.split(". ") if part.strip()]

    def _extract_shared_video_theme(self, master_prompt: str) -> str:
        sentences = self._split_prompt_sentences(master_prompt)
        if not sentences:
            return master_prompt.strip()

        shared = [
            sentence.rstrip(".")
            for sentence in sentences
            if not any(
                marker in sentence.lower()
                for marker in (
                    "scene ",
                    "start ",
                    "begin ",
                    "pull back",
                    "pan ",
                    "cut to",
                    "close on",
                    "end with",
                    "opening shot",
                    "final shot",
                    "camera move",
                )
            )
        ]
        selected = shared[:3] if shared else [sentence.rstrip(".") for sentence in sentences[:2]]
        return ". ".join(selected).strip()

    def _build_scene_prompt(
        self,
        *,
        master_prompt: str,
        shot: str,
        scene_index: int,
        total_scenes: int,
        concept_name: str,
        opening_shot: str,
        end_frame: str,
    ) -> str:
        shared_theme = self._extract_shared_video_theme(master_prompt)
        if total_scenes <= 1:
            stage_direction = f"Build the full scene around {shot}."
        elif scene_index == 1:
            stage_direction = f"Use this scene to establish the hook with {opening_shot or shot}."
        elif scene_index == total_scenes:
            stage_direction = f"Use this scene to land the payoff and transition cleanly into {end_frame or shot}."
        else:
            stage_direction = f"Use this scene to advance the story with a focused beat around {shot}."

        return " ".join(
            part
            for part in [
                f"Create scene {scene_index} of {total_scenes} for {concept_name}." if concept_name else "",
                f"Shared creative direction: {shared_theme}." if shared_theme else "",
                f"This scene should focus specifically on: {shot}.",
                stage_direction,
                "Maintain the same subject identity, styling, lighting continuity, and brand cues.",
                "Keep the framing optimized for short-form paid social performance.",
            ]
            if part
        )

    async def persist_artifact(
        self,
        *,
        brand_id: str,
        storage_prefix: str,
        artifact: RenderedBinary,
    ) -> RenderedBinary:
        key = self._storage.build_key(
            brand_id,
            f"{storage_prefix.strip('/').replace('/', '-')}-{artifact.file_name}",
        )
        stored_url = await self._storage.save_asset(
            key=key,
            data=artifact.data,
            content_type=artifact.mime_type,
        )
        artifact.stored_url = stored_url
        return artifact

    async def _render_scene_or_extension(
        self,
        *,
        scene: dict[str, Any],
        scene_index: int,
        model_id: str,
        settings: dict[str, Any],
        selected_assets: list[dict[str, Any]],
        storage_prefix: str,
        previous_video_gcs_uri: str | None,
    ) -> dict[str, Any]:
        mode = self._resolve_generation_mode(
            settings=settings,
            scene=scene,
            selected_assets=selected_assets,
            has_previous=previous_video_gcs_uri is not None,
        )
        scene_logs = [f"Scene {scene_index + 1}: using {mode.replace('_', ' ')} generation mode."]
        parameters = self._build_generation_parameters(
            settings=settings,
            storage_prefix=storage_prefix,
            scene=scene,
        )

        if mode == "extend" and previous_video_gcs_uri is not None:
            operation = await self._veo.extend(
                prompt=str(scene.get("prompt") or ""),
                video_gcs_uri=previous_video_gcs_uri,
                model_id=model_id,
                parameters=parameters,
            )
        else:
            image, last_frame, reference_images = await self._resolve_scene_inputs(
                mode=mode,
                scene=scene,
                settings=settings,
                selected_assets=selected_assets,
            )
            operation = await self._veo.generate(
                prompt=str(scene.get("prompt") or ""),
                model_id=model_id,
                parameters=parameters,
                image=image,
                last_frame=last_frame,
                reference_images=reference_images,
            )

        operation_name = operation.get("name")
        if not operation_name:
            raise VideoPipelineError("Veo did not return an operation name.")
        scene_logs.append(f"Scene {scene_index + 1}: operation started {operation_name}.")

        result = await self._veo.poll_operation(
            operation_name=str(operation_name),
            model_id=model_id,
        )
        video_payload = self._veo.extract_video_payload(result)
        video_gcs_uri = video_payload.get("gcs_uri")
        if video_gcs_uri:
            video_bytes = await self._veo.download_gcs_uri(str(video_gcs_uri))
            mime_type = str(video_payload.get("mime_type") or "video/mp4")
            source_note = "downloaded from GCS"
        else:
            inline_bytes = video_payload.get("bytes")
            if not isinstance(inline_bytes, bytes) or not inline_bytes:
                raise VideoPipelineError("Veo returned inline video data, but it could not be decoded.")
            video_bytes = inline_bytes
            mime_type = str(video_payload.get("mime_type") or "video/mp4")
            source_note = "decoded from inline response bytes"
        artifact = RenderedBinary(
            kind="scene_video",
            label=str(scene.get("title") or f"Scene {scene_index + 1}"),
            data=video_bytes,
            mime_type=mime_type,
            file_name=f"scene-{scene_index + 1:02d}.mp4",
            source_gcs_uri=str(video_gcs_uri) if video_gcs_uri else None,
            scene_id=str(scene.get("id") or f"scene-{scene_index + 1}"),
        )
        scene_logs.append(f"Scene {scene_index + 1}: render completed and video was {source_note}.")
        return {"artifact": artifact, "video_gcs_uri": str(video_gcs_uri) if video_gcs_uri else None, "logs": scene_logs}

    def _build_generation_parameters(
        self,
        *,
        settings: dict[str, Any],
        storage_prefix: str,
        scene: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "storageUri": self._veo.storage_uri(storage_prefix),
            "aspectRatio": settings.get("aspect_ratio") or "9:16",
            "resolution": settings.get("resolution") or "1080p",
            "sampleCount": 1,
        }
        if settings.get("render_strategy") != "daisy_chain":
            params["durationSeconds"] = int(
                scene.get("duration_seconds") or settings.get("scene_duration_seconds") or 8
            )
        else:
            params["durationSeconds"] = int(
                scene.get("duration_seconds") or settings.get("scene_duration_seconds") or 8
            )
        if settings.get("negative_prompt"):
            params["negativePrompt"] = str(settings["negative_prompt"])
        if settings.get("seed") is not None:
            params["seed"] = int(settings["seed"])
        if settings.get("generate_native_audio"):
            params["generateAudio"] = True
        if settings.get("model_id", "").endswith("preview"):
            params["compressionQuality"] = "optimized"
        return params

    def _resolve_generation_mode(
        self,
        *,
        settings: dict[str, Any],
        scene: dict[str, Any],
        selected_assets: list[dict[str, Any]],
        has_previous: bool,
    ) -> str:
        if has_previous and settings.get("render_strategy") == "daisy_chain":
            return "extend"

        requested = str(settings.get("generation_mode") or "auto")
        if requested != "auto":
            return requested

        scene_ref_ids = {str(asset_id) for asset_id in scene.get("reference_asset_ids") or []}
        image_assets = [
            asset
            for asset in selected_assets
            if str(asset.get("mime_type") or "").startswith("image/")
            and (not scene_ref_ids or str(asset.get("id")) in scene_ref_ids)
        ]
        last_frame_asset_id = str(settings.get("last_frame_asset_id") or "").strip()
        if last_frame_asset_id and image_assets:
            return "first_last_frame"
        if len(image_assets) >= 2:
            return "reference_images"
        if len(image_assets) >= 1:
            return "image_to_video"
        return "prompt_only"

    async def _resolve_scene_inputs(
        self,
        *,
        mode: str,
        scene: dict[str, Any],
        settings: dict[str, Any],
        selected_assets: list[dict[str, Any]],
    ) -> tuple[dict[str, str] | None, dict[str, str] | None, list[dict[str, Any]] | None]:
        scene_ref_ids = {str(asset_id) for asset_id in scene.get("reference_asset_ids") or []}
        image_assets = [
            asset
            for asset in selected_assets
            if str(asset.get("mime_type") or "").startswith("image/")
            and (not scene_ref_ids or str(asset.get("id")) in scene_ref_ids)
        ]

        if mode == "prompt_only":
            return None, None, None

        if mode == "reference_images":
            reference_images = []
            for asset in image_assets[:3]:
                reference_images.append(
                    {
                        "image": await self._asset_to_vertex_image(asset),
                        "referenceType": "asset",
                    }
                )
            if not reference_images:
                raise VideoPipelineError("Reference-images mode requires at least one image asset.")
            return None, None, reference_images

        if mode == "first_last_frame":
            if not image_assets:
                raise VideoPipelineError("First/last-frame mode requires at least one image asset.")
            first_frame_asset_id = str(settings.get("first_frame_asset_id") or "").strip()
            ordered_assets = image_assets
            if first_frame_asset_id:
                ordered_assets = sorted(
                    image_assets,
                    key=lambda asset: 0 if str(asset.get("id")) == first_frame_asset_id else 1,
                )
            image = await self._asset_to_vertex_image(ordered_assets[0])
            last_frame = None
            last_frame_asset_id = str(settings.get("last_frame_asset_id") or "").strip()
            for asset in ordered_assets[1:]:
                if not last_frame_asset_id or str(asset.get("id")) == last_frame_asset_id:
                    last_frame = await self._asset_to_vertex_image(asset)
                    break
            return image, last_frame, None

        if mode == "image_to_video":
            if not image_assets:
                raise VideoPipelineError("Image-to-video mode requires at least one image asset.")
            image = await self._asset_to_vertex_image(image_assets[0])
            return image, None, None

        raise VideoPipelineError(f"Unsupported generation mode: {mode}")

    async def _asset_to_vertex_image(self, asset: dict[str, Any]) -> dict[str, str]:
        binary = await self._load_asset_bytes(asset)
        mime_type = str(asset.get("mime_type") or "image/png")
        return {
            "bytesBase64Encoded": base64.b64encode(binary).decode("ascii"),
            "mimeType": mime_type,
        }

    async def _load_asset_bytes(self, asset: dict[str, Any]) -> bytes:
        stored_url = str(asset.get("stored_url") or "").strip()
        source_url = str(asset.get("source_url") or "").strip()
        mime_type = str(asset.get("mime_type") or "")

        if stored_url:
            data, _ = await self._storage.read_asset(stored_url, fallback_content_type=mime_type or None)
            return data

        if source_url.startswith("http://") or source_url.startswith("https://"):
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(source_url)
                response.raise_for_status()
                return response.content

        raise VideoPipelineError(f"Asset {asset.get('id')} is missing a readable stored/source URL.")

    def _default_music_prompt(self, *, brand_name: str, execution: dict[str, Any]) -> str:
        concept_name = str(execution.get("concept_name") or "campaign")
        angle = str(execution.get("video_brief", {}).get("concept") or concept_name)
        return (
            f"Cinematic, modern short-form ad soundtrack for {brand_name}. "
            f"Support the concept '{angle}' with polished commercial energy, strong build, "
            "and no vocals."
        )
