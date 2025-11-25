"""
GroqClient (Free Tier) - Updated Nov 2025
Drop-in replacement for HFInferenceClient using Groq's free tier.
Now with model validation to avoid 404 errors.
"""

import time
import requests
from typing import List, Dict, Optional

from dotenv import load_dotenv
import os

load_dotenv()  # reads .env automatically

groq_api_key = os.getenv("GROQ_API_KEY")

try:
    from src.utils.logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GroqClient")





class GroqClient:
    SUPPORTED_MODELS = [  # Latest as of Nov 2025
        "llama-3.1-70b-versatile", "llama3-70b-8192", "llama3-8b-8192",
        "gemma2-9b-it", "whisper-large-v3-turbo", "llama-guard-3-8b"
    ]

    def __init__(
        self,
        model: str = "llama-3.1-70b-versatile",  # Default to a real supported model
        api_key: str = groq_api_key,
        max_retries: int = 5,
        retry_backoff: float = 1.0,
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key.rstrip()
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout

        if not self.api_key or not self.api_key.startswith("gsk_"):
            raise ValueError(
                "Groq API key required! Get your free key at:\n"
                "https://console.groq.com/keys"
            )

        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.models_url = "https://api.groq.com/openai/v1/models"  # For validation
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Validate model on init
        self._validate_model()

        logger.info(f"[GroqClient] Initialized with model: {self.model}")

    def _validate_model(self):
        """Check if model is supported via /models endpoint."""
        try:
            response = requests.get(self.models_url, headers={"Authorization": self.headers["Authorization"]}, timeout=10)
            response.raise_for_status()
            available_models = [m["id"] for m in response.json().get("data", [])]

            if self.model not in available_models:
                suggestions = [m for m in self.SUPPORTED_MODELS if m in available_models][:3]
                raise ValueError(
                    f"Model '{self.model}' not supported on Groq (404 error cause).\n"
                    f"Available: {', '.join(available_models[:5])}...\n"
                    f"Try one of these: {', '.join(suggestions) or 'llama-3.1-70b-versatile'}"
                )
        except Exception as e:
            logger.warning(f"Model validation failed (non-fatal): {e}. Proceeding anyway.")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "stream": False
        }

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 404:
                    # Specific handling for model 404
                    raise ValueError(
                        f"Model '{self.model}' not found (404). "
                        f"Use a supported model like 'llama-3.1-70b-versatile'."
                    )

                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit hit. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                return content.strip()

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"[GroqClient] Attempt {attempt} failed: {e}")
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

            except Exception as e:
                last_error = e
                logger.warning(f"[GroqClient] Unexpected error (attempt {attempt}): {e}")
                time.sleep(self.retry_backoff * (2 ** (attempt - 1)))

        logger.error(f"[GroqClient] All {self.max_retries} attempts failed: {last_error}")
        raise ConnectionError(f"Groq API unreachable after {self.max_retries} retries") from last_error