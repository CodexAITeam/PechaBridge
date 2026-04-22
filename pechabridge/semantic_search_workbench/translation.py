"""OpenAI-backed translation bridge for cross-lingual search."""

from __future__ import annotations

import os

from openai import OpenAI

from .config import TranslationConfig


class TranslationProxy:
    """Translates search queries into Classical Tibetan and back to English."""

    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self._client: OpenAI | None = None
        if not config.enabled:
            return
        if config.provider.lower() != "openai":
            raise ValueError(f"Unsupported translation provider: {config.provider}")

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env} is required for OpenAI translation."
            )

        client_kwargs = {"api_key": api_key}
        if config.base_url_env:
            base_url = os.getenv(config.base_url_env)
            if base_url:
                client_kwargs["base_url"] = base_url
        if config.organization_env:
            organization = os.getenv(config.organization_env)
            if organization:
                client_kwargs["organization"] = organization

        self._client = OpenAI(**client_kwargs)

    def translate_query_to_tibetan(self, query: str) -> str:
        if not self.config.enabled:
            return query
        return self._generate(
            instructions=self.config.translate_query_prompt,
            input_text=query,
        )

    def back_translate_to_english(self, context: str) -> str:
        if not self.config.enabled:
            return context
        return self._generate(
            instructions=self.config.back_translate_prompt,
            input_text=context,
        )

    def _generate(self, instructions: str, input_text: str) -> str:
        if self._client is None:
            return input_text

        if hasattr(self._client, "responses"):
            response = self._client.responses.create(
                model=self.config.model,
                instructions=instructions,
                input=input_text,
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
            )
            output_text = getattr(response, "output_text", "") or ""
            if output_text.strip():
                return output_text.strip()

            chunks: list[str] = []
            for output_item in getattr(response, "output", []):
                for content_item in getattr(output_item, "content", []):
                    text_value = getattr(content_item, "text", None)
                    if text_value:
                        chunks.append(text_value)
            return "\n".join(chunks).strip()

        completion = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
