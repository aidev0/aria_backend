from __future__ import annotations

import io
import os
import wave
from typing import Any

import openai


class WhisperService:
    """OpenAI Whisper speech-to-text service for processing glasses microphone audio."""

    def __init__(self):
        self._client = None
        self._audio_buffer: bytearray = bytearray()
        self._sample_rate = 16000
        self._channels = 1
        self._sample_width = 2  # 16-bit audio

    @property
    def client(self):
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"))
        return self._client

    async def transcribe(
        self,
        audio_data: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio data using OpenAI Whisper.

        Args:
            audio_data: Raw audio bytes or WAV data
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional prompt to guide transcription
        """
        # Wrap raw PCM in WAV format if needed
        if not audio_data[:4] == b"RIFF":
            audio_data = self._pcm_to_wav(audio_data)

        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"

        kwargs: dict[str, Any] = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        response = await self.client.audio.transcriptions.create(**kwargs)

        return {
            "text": response.text,
            "language": getattr(response, "language", language),
            "segments": [
                {
                    "text": seg.text if hasattr(seg, 'text') else seg.get("text", ""),
                    "start": seg.start if hasattr(seg, 'start') else seg.get("start", 0),
                    "end": seg.end if hasattr(seg, 'end') else seg.get("end", 0),
                }
                for seg in (getattr(response, "segments", None) or [])
            ],
        }

    def accumulate_audio(self, chunk: bytes) -> None:
        """Accumulate audio chunks from the glasses microphone stream."""
        self._audio_buffer.extend(chunk)

    def flush_buffer(self) -> bytes | None:
        """Flush the audio buffer and return accumulated audio as WAV."""
        if not self._audio_buffer:
            return None
        wav_data = self._pcm_to_wav(bytes(self._audio_buffer))
        self._audio_buffer.clear()
        return wav_data

    async def transcribe_stream(self, min_duration_ms: int = 2000) -> dict[str, Any] | None:
        """Transcribe accumulated audio if buffer has enough data.

        Args:
            min_duration_ms: Minimum audio duration in ms before transcribing

        Returns:
            Dict with text, language, segments, and wav_data (raw WAV bytes for storage)
        """
        min_bytes = int(
            self._sample_rate * self._sample_width * self._channels * min_duration_ms / 1000
        )
        if len(self._audio_buffer) < min_bytes:
            return None

        wav_data = self.flush_buffer()
        if wav_data:
            result = await self.transcribe(wav_data, language="en")
            result["wav_data"] = wav_data
            duration_ms = int(len(wav_data) / (self._sample_rate * self._sample_width * self._channels) * 1000)
            result["duration_ms"] = duration_ms
            return result
        return None

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM audio data to WAV format."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(self._sample_width)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()
