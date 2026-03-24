from __future__ import annotations

import os


class AIService:
    """Unified AI service supporting Claude, Gemini, and OpenAI models."""

    def __init__(self):
        self._claude_client = None
        self._gemini_model = None
        self._openai_client = None

    @property
    def claude(self):
        if self._claude_client is None:
            import anthropic

            self._claude_client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        return self._claude_client

    @property
    def openai(self):
        if self._openai_client is None:
            import openai

            self._openai_client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        return self._openai_client

    @property
    def gemini(self):
        if self._gemini_model is None:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        return self._gemini_model

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = "claude",
        max_tokens: int = 8192,
    ) -> str:
        """Generate a response using the specified model."""
        if model == "claude":
            return await self._generate_claude(prompt, system_prompt, max_tokens)
        elif model == "gemini":
            return await self._generate_gemini(prompt, system_prompt)
        elif model in ("openai", "codex", "gpt"):
            return await self._generate_openai(prompt, system_prompt, max_tokens)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'claude', 'gemini', or 'openai'.")

    async def _generate_claude(
        self, prompt: str, system_prompt: str, max_tokens: int
    ) -> str:
        response = await self.claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _generate_gemini(self, prompt: str, system_prompt: str) -> str:
        import asyncio

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = await asyncio.to_thread(
            self.gemini.generate_content, full_prompt
        )
        return response.text

    async def _generate_openai(
        self, prompt: str, system_prompt: str, max_tokens: int
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def generate_with_vision(
        self,
        prompt: str,
        image_base64: str,
        system_prompt: str = "",
        model: str = "claude",
    ) -> str:
        """Generate a response with image input (for glasses camera frames)."""
        if model == "claude":
            response = await self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.content[0].text
        elif model == "openai":
            response = await self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=4096,
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Vision not supported for model: {model}")
