from __future__ import annotations

import os
from typing import AsyncIterator

import httpx


class ElevenLabsService:
    """ElevenLabs text-to-speech service for generating voice memos and glasses whisper."""

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.default_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
        self.model_id = "eleven_turbo_v2_5"

    def _headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def text_to_speech(
        self,
        text: str,
        voice_id: str | None = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
    ) -> bytes:
        """Convert text to speech audio bytes (MP3).

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID (defaults to Rachel)
            stability: Voice stability (0-1)
            similarity_boost: Voice similarity boost (0-1)
            style: Style exaggeration (0-1)

        Returns:
            MP3 audio bytes
        """
        voice = voice_id or self.default_voice_id
        url = f"{self.BASE_URL}/text-to-speech/{voice}"

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            return response.content

    async def text_to_speech_stream(
        self,
        text: str,
        voice_id: str | None = None,
        chunk_size: int = 4096,
    ) -> AsyncIterator[bytes]:
        """Stream text-to-speech audio chunks for real-time playback on glasses.

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            chunk_size: Size of audio chunks to yield

        Yields:
            Audio bytes chunks (MP3)
        """
        voice = voice_id or self.default_voice_id
        url = f"{self.BASE_URL}/text-to-speech/{voice}/stream"

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size):
                    yield chunk

    async def list_voices(self) -> list[dict]:
        """List available ElevenLabs voices."""
        url = f"{self.BASE_URL}/voices"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            return [
                {
                    "voice_id": v["voice_id"],
                    "name": v["name"],
                    "category": v.get("category", ""),
                }
                for v in data.get("voices", [])
            ]
