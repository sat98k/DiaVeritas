# =============================================================================
# src/generation/llm_client.py
#
# Multi-provider LLM client for DiaVeritas.
#
# Supports:
#   - Groq (default: llama-3.3-70b-versatile, free tier, fast)
#   - OpenAI (GPT-4o, GPT-4, etc.)
#   - Google Gemini
#   - Anthropic Claude
#
# Provider is set via LLM_PROVIDER in .env.
# Groq and OpenAI share the OpenAI-compatible client interface.
#
# If no API key is set, the client will raise a clear error
# rather than silently returning empty output.
#
# The LLM is used for:
#   1. Structured claim extraction (claim_extractor.py)
#   2. Final answer synthesis (synthesizer.py)
#
# IMPORTANT: The LLM synthesizes analyzed evidence.
# It does NOT independently decide medical truth.
# =============================================================================

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config import settings


class LLMClient:
    """
    Unified LLM client that dispatches to the configured provider.

    Usage:
        client = LLMClient()
        response = client.complete("Your prompt here")
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self._provider = (provider or settings.llm_provider).lower()
        self._model = model or settings.llm_model
        self._client = None
        logger.info(f"LLM: provider={self._provider}, model={self._model}")

    def _load(self):
        """Initialize the provider-specific client on first use."""
        if self._client is not None:
            return

        if self._provider in ("groq", "openai"):
            self._load_openai_compatible()
        elif self._provider == "anthropic":
            self._load_anthropic()
        elif self._provider == "google":
            self._load_google()
        else:
            raise ValueError(
                f"Unknown LLM provider: '{self._provider}'. "
                "Set LLM_PROVIDER to: openai | groq | google | anthropic"
            )

    def _load_openai_compatible(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: py -m pip install openai")

        if self._provider == "groq":
            api_key = settings.groq_api_key
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not set. Add it to your .env file. "
                    "Get a free key at: https://console.groq.com/"
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:  # openai
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set.")
            self._client = OpenAI(api_key=api_key)

    def _load_anthropic(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: py -m pip install anthropic")
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def _load_google(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required: py -m pip install google-generativeai"
            )
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=settings.google_api_key)
        self._client = genai.GenerativeModel(self._model)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system message (used where supported).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. Lower = more deterministic.

        Returns:
            The generated text as a string.

        Raises:
            ValueError: If API key is not configured.
            RuntimeError: If the API call fails.
        """
        self._load()

        try:
            if self._provider in ("groq", "openai"):
                return self._complete_openai(prompt, system_prompt, max_tokens, temperature)
            elif self._provider == "anthropic":
                return self._complete_anthropic(prompt, system_prompt, max_tokens, temperature)
            elif self._provider == "google":
                return self._complete_google(prompt, max_tokens, temperature)
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed (provider={self._provider}): {e}"
            ) from e

    def _complete_openai(self, prompt, system_prompt, max_tokens, temperature) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _complete_anthropic(self, prompt, system_prompt, max_tokens, temperature) -> str:
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()

    def _complete_google(self, prompt, max_tokens, temperature) -> str:
        response = self._client.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return response.text.strip()

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model
