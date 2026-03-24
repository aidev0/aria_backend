from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class StreamFrame:
    """A single video frame from glasses."""
    data: str  # base64-encoded JPEG
    timestamp: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0


@dataclass
class AudioChunk:
    """A chunk of audio data from glasses mic."""
    data: bytes
    timestamp: float = field(default_factory=time.time)
    sample_rate: int = 16000


class StreamService:
    """Manages video and audio streams between glasses, AI, and display outputs."""

    def __init__(self):
        self._video_subscribers: list[Callable] = []
        self._audio_subscribers: list[Callable] = []
        self._ai_response_subscribers: list[Callable] = []
        self._latest_frame: StreamFrame | None = None
        self._frame_count = 0
        self._stream_active = False
        self._audio_active = False

    @property
    def is_streaming(self) -> bool:
        return self._stream_active

    @property
    def latest_frame(self) -> StreamFrame | None:
        return self._latest_frame

    # --- Video Stream ---

    def subscribe_video(self, callback: Callable[[StreamFrame], Coroutine]) -> None:
        """Subscribe to video frames from glasses."""
        self._video_subscribers.append(callback)

    def unsubscribe_video(self, callback: Callable) -> None:
        self._video_subscribers = [cb for cb in self._video_subscribers if cb != callback]

    async def push_video_frame(self, frame_data: str) -> None:
        """Push a video frame from glasses to all subscribers.

        Args:
            frame_data: Base64-encoded JPEG frame
        """
        frame = StreamFrame(data=frame_data)
        self._latest_frame = frame
        self._frame_count += 1
        self._stream_active = True

        tasks = [cb(frame) for cb in self._video_subscribers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # --- Audio Stream ---

    def subscribe_audio(self, callback: Callable[[AudioChunk], Coroutine]) -> None:
        """Subscribe to audio chunks from glasses mic."""
        self._audio_subscribers.append(callback)

    def unsubscribe_audio(self, callback: Callable) -> None:
        self._audio_subscribers = [cb for cb in self._audio_subscribers if cb != callback]

    async def push_audio_chunk(self, audio_data: bytes, sample_rate: int = 16000) -> None:
        """Push an audio chunk from glasses mic to all subscribers.

        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Audio sample rate
        """
        chunk = AudioChunk(data=audio_data, sample_rate=sample_rate)
        self._audio_active = True

        tasks = [cb(chunk) for cb in self._audio_subscribers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # --- AI Response Stream (to glasses display + speaker) ---

    def subscribe_ai_response(self, callback: Callable[[dict], Coroutine]) -> None:
        """Subscribe to AI responses for glasses display/speaker output."""
        self._ai_response_subscribers.append(callback)

    async def push_ai_response(
        self,
        text: str,
        audio_data: bytes | None = None,
        display_data: dict | None = None,
    ) -> None:
        """Push AI response to glasses display and speaker.

        Args:
            text: AI response text
            audio_data: Optional TTS audio bytes for speaker
            display_data: Optional structured data for glasses display
        """
        response = {
            "type": "ai_response",
            "text": text,
            "timestamp": time.time(),
        }

        if audio_data:
            response["audio"] = base64.b64encode(audio_data).decode("utf-8")

        if display_data:
            response["display"] = display_data

        tasks = [cb(response) for cb in self._ai_response_subscribers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # --- Stream Control ---

    def start_video(self) -> None:
        self._stream_active = True
        self._frame_count = 0

    def stop_video(self) -> None:
        self._stream_active = False
        self._latest_frame = None

    def start_audio(self) -> None:
        self._audio_active = True

    def stop_audio(self) -> None:
        self._audio_active = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "video_streaming": self._stream_active,
            "audio_active": self._audio_active,
            "frames_received": self._frame_count,
            "video_subscribers": len(self._video_subscribers),
            "audio_subscribers": len(self._audio_subscribers),
            "ai_response_subscribers": len(self._ai_response_subscribers),
        }
