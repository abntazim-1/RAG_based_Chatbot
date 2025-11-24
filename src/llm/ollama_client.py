# src/llm/ollama_client.py
"""
OllamaClient
-------------
A lightweight, production-ready client for interacting with a local Ollama server.

Responsibilities:
- Send text or chat-based prompts to the local Ollama API.
- Handle retries, logging, and tolerant response parsing.
- Designed for integration in RAG pipelines with minimal dependencies.

Example:
    client = OllamaClient(model="gemma:2b")
    response = client.generate("Explain RAG in one sentence.")
    print(response)
"""

from __future__ import annotations
import json
import time
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

try:
    from src.utils.logger import logger
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("OllamaClient")


@dataclass
class OllamaClient:
    model: str
    host: str = "http://localhost:11434"
    timeout: int = 120
    max_retries: int = 3
    retry_backoff: float = 0.5

    def _post(self, endpoint: str, payload: Dict[str, Any], stream: bool = False) -> Dict[str, Any]:
        """Send POST request to Ollama server with retry logic."""
        url = f"{self.host.rstrip('/')}/{endpoint.lstrip('/')}"
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Add stream parameter to payload if not already present
                if "stream" not in payload:
                    payload["stream"] = stream
                
                resp = requests.post(url, json=payload, timeout=self.timeout, stream=stream)
                
                # Check for 404 specifically and provide helpful error
                if resp.status_code == 404:
                    # Try to get more info from the response
                    try:
                        error_detail = resp.json().get("error", "Unknown error")
                    except:
                        error_detail = resp.text[:200] if resp.text else "No error details"
                    
                    error_msg = (
                        f"Ollama API endpoint not found: {url}\n"
                        f"Status: {resp.status_code}\n"
                        f"Error: {error_detail}\n\n"
                        f"This usually means:\n"
                        f"  1. Ollama is not running - Start it with: ollama serve\n"
                        f"  2. Ollama is running on a different port - Check with: ollama list\n"
                        f"  3. The model '{payload.get('model', 'unknown')}' is not installed - Install with: ollama pull {payload.get('model', 'model-name')}\n"
                        f"  4. Check if Ollama is accessible: curl {self.host}/api/tags"
                    )
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                
                resp.raise_for_status()
                
                if stream:
                    # Handle streaming response (NDJSON format - one JSON per line)
                    full_response = {"response": ""}
                    for line in resp.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if "response" in chunk:
                                    full_response["response"] += chunk["response"]
                                # Keep the last chunk's metadata
                                if chunk.get("done", False):
                                    full_response.update({k: v for k, v in chunk.items() if k != "response"})
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response
                else:
                    # Non-streaming response - try to parse as single JSON
                    try:
                        return resp.json()
                    except json.JSONDecodeError:
                        # Handle case where Ollama returns streaming even when stream=False
                        # Parse as NDJSON (one JSON per line)
                        content = resp.text.strip()
                        if '\n' in content:
                            # Multiple JSON objects - combine responses
                            full_response = {"response": ""}
                            for line in content.split('\n'):
                                if line.strip():
                                    try:
                                        chunk = json.loads(line)
                                        if "response" in chunk:
                                            full_response["response"] += chunk["response"]
                                        if chunk.get("done", False):
                                            full_response.update({k: v for k, v in chunk.items() if k != "response"})
                                            break
                                    except json.JSONDecodeError:
                                        continue
                            return full_response
                        else:
                            # Single line but still failed - re-raise
                            raise
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"[OllamaClient] JSON decode error on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt == 1:
                    # On first connection error, provide helpful diagnostics
                    logger.error(f"❌ Cannot connect to Ollama at {self.host}")
                    logger.error("   Please verify:")
                    logger.error(f"   1. Ollama is running: ollama serve")
                    logger.error(f"   2. Ollama is accessible: curl {self.host}/api/tags")
                    logger.error(f"   3. Check if port is correct (default: 11434)")
                logger.warning(
                    f"[OllamaClient] Attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[OllamaClient] Attempt {attempt}/{self.max_retries} failed: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        logger.error(f"[OllamaClient] Request failed after {self.max_retries} retries: {last_error}")
        raise last_error

    def _parse_response(self, data: Dict[str, Any]) -> str:
        """Extract response text safely from Ollama's response format."""
        if not data:
            return ""
        if "response" in data:
            return str(data["response"]).strip()
        if "message" in data and isinstance(data["message"], dict):
            return str(data["message"].get("content", "")).strip()
        return json.dumps(data, ensure_ascii=False)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """Generate text completion from a prompt."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,  # Explicitly set stream parameter
        }
        if stop:
            payload["stop"] = stop

        logger.debug(f"[OllamaClient] Generating response for prompt length={len(prompt)}")
        data = self._post("/api/generate", payload, stream=stream)
        return self._parse_response(data)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        """Send chat-style conversation and return assistant reply."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,  # Explicitly set stream parameter
        }

        try:
            data = self._post("/api/chat", payload, stream=stream)
        except Exception:
            # fallback to prompt-based generation if chat endpoint unavailable
            prompt = self._messages_to_prompt(messages)
            generate_payload = {**payload, "prompt": prompt}
            data = self._post("/api/generate", generate_payload, stream=stream)
        return self._parse_response(data)

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Convert chat messages into a single prompt string."""
        lines = [f"{m['role']}: {m['content']}" for m in messages if "content" in m]
        lines.append("assistant:")
        return "\n".join(lines)


if __name__ == "__main__":
    client = OllamaClient(model="gemma:2b")
    try:
        out = client.generate("Give a short definition of Retrieval-Augmented Generation (RAG).")
        print("Response:", out)
    except Exception as e:
        logger.error(f"Error during test run: {e}")
