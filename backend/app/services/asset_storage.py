"""Storage abstraction for local disk and Cloudflare R2-backed brand assets."""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import settings


class AssetStorage:
    """Read and write assets using local disk in dev and R2 in production."""

    def __init__(
        self,
        *,
        storage_dir: str = "assets",
        r2_endpoint: str = "",
        r2_access_key: str = "",
        r2_secret_key: str = "",
        r2_bucket: str = "sloth-assets",
    ) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._r2_bucket = r2_bucket
        self._r2_enabled = all([r2_endpoint, r2_access_key, r2_secret_key, r2_bucket])
        self._r2_client = None
        if self._r2_enabled:
            self._r2_client = boto3.client(
                "s3",
                endpoint_url=r2_endpoint,
                aws_access_key_id=r2_access_key,
                aws_secret_access_key=r2_secret_key,
                region_name="auto",
                config=Config(signature_version="s3v4"),
            )

    @classmethod
    def from_settings(cls) -> "AssetStorage":
        return cls(
            storage_dir="assets",
            r2_endpoint=settings.R2_ENDPOINT,
            r2_access_key=settings.R2_ACCESS_KEY,
            r2_secret_key=settings.R2_SECRET_KEY,
            r2_bucket=settings.R2_BUCKET,
        )

    @property
    def r2_enabled(self) -> bool:
        return self._r2_enabled

    def build_key(self, brand_id: str, file_name: str) -> str:
        return f"assets/{brand_id}/{file_name}"

    async def save_asset(
        self,
        *,
        key: str,
        data: bytes,
        content_type: str,
    ) -> str:
        if self._r2_enabled and self._r2_client is not None:
            await asyncio.to_thread(
                self._r2_client.put_object,
                Bucket=self._r2_bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return key

        path = self._storage_dir.parent / key if not key.startswith(str(self._storage_dir)) else Path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)
        return key

    async def read_asset(
        self,
        stored_url: str,
        fallback_content_type: str | None = None,
    ) -> tuple[bytes, str]:
        key = stored_url.lstrip("/")
        local_path = self._storage_dir.parent / key if not key.startswith(str(self._storage_dir)) else Path(key)

        if local_path.exists():
            data = await asyncio.to_thread(local_path.read_bytes)
            return data, fallback_content_type or self._guess_content_type(key)

        if self._r2_enabled and self._r2_client is not None:
            try:
                response = await asyncio.to_thread(
                    self._r2_client.get_object,
                    Bucket=self._r2_bucket,
                    Key=key,
                )
            except ClientError as exc:
                raise FileNotFoundError(key) from exc

            body = response["Body"]
            data = await asyncio.to_thread(body.read)
            content_type = response.get("ContentType") or fallback_content_type or self._guess_content_type(key)
            return data, content_type

        raise FileNotFoundError(key)

    @staticmethod
    def _guess_content_type(key: str) -> str:
        return mimetypes.guess_type(key)[0] or "application/octet-stream"
