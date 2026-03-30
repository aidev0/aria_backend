from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

from google.cloud import storage


class GCSService:
    """Google Cloud Storage service for storing voice audio recordings."""

    def __init__(self):
        self._client = None
        self._bucket_name = os.getenv("GCS_BUCKET_NAME", "aria-voice-recordings")

    @property
    def client(self) -> storage.Client:
        if self._client is None:
            self._client = storage.Client()
        return self._client

    @property
    def bucket(self) -> storage.Bucket:
        return self.client.bucket(self._bucket_name)

    def upload_audio_sync(
        self,
        audio_data: bytes,
        content_type: str = "audio/wav",
        session_id: str | None = None,
    ) -> dict:
        """Upload audio data to GCS (blocking). Use via asyncio.to_thread().

        Returns:
            Dict with gcs_uri, public_url, blob_name, size_bytes
        """
        if session_id is None:
            session_id = uuid.uuid4().hex

        timestamp = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        blob_name = f"voice/{timestamp}/{session_id}.wav"

        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(audio_data, content_type=content_type)

        return {
            "gcs_uri": f"gs://{self._bucket_name}/{blob_name}",
            "public_url": blob.public_url,
            "blob_name": blob_name,
            "size_bytes": len(audio_data),
        }

    async def upload_audio(
        self,
        audio_data: bytes,
        content_type: str = "audio/wav",
        session_id: str | None = None,
    ) -> dict:
        """Async wrapper around upload_audio_sync."""
        return await asyncio.to_thread(
            self.upload_audio_sync, audio_data, content_type, session_id
        )

    def get_signed_url(self, blob_name: str, expiration_minutes: int = 60) -> str:
        """Generate a signed URL for accessing a stored audio file."""
        blob = self.bucket.blob(blob_name)
        return blob.generate_signed_url(
            expiration=timedelta(minutes=expiration_minutes),
            method="GET",
        )
