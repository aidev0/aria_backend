from __future__ import annotations

import os

from twilio.rest import Client as TwilioClient


class WhatsAppService:
    """WhatsApp messaging service via Twilio for sending pipeline reports."""

    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
        self._client: TwilioClient | None = None

    @property
    def client(self) -> TwilioClient:
        if self._client is None:
            self._client = TwilioClient(self.account_sid, self.auth_token)
        return self._client

    async def send_message(self, to: str, message: str) -> dict:
        """Send a WhatsApp message.

        Args:
            to: Recipient phone number (e.g., '+1234567890')
            message: Message body text

        Returns:
            Message delivery status dict
        """
        import asyncio

        if not to.startswith("whatsapp:"):
            to = f"whatsapp:{to}"

        result = await asyncio.to_thread(
            self._send_sync, to, message
        )
        return result

    def _send_sync(self, to: str, message: str) -> dict:
        msg = self.client.messages.create(
            body=message,
            from_=self.from_number,
            to=to,
        )
        return {
            "sid": msg.sid,
            "status": msg.status,
            "to": to,
        }

    async def send_report(self, to: str, report: dict) -> dict:
        """Send a formatted pipeline report via WhatsApp.

        Args:
            to: Recipient phone number
            report: Pipeline report dict from ReporterAgent
        """
        message = report.get("whatsapp_message", "")
        if not message:
            # Build message from report details
            status = report.get("status", "unknown")
            summary = report.get("summary", "No summary available")
            message = f"*Aria Pipeline Report*\n\nStatus: {status}\n{summary}"

        return await self.send_message(to, message)
