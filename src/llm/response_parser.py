"""
response_parser.py

High-performance, minimal, and modular response parser for RAG or LLM-based pipelines.
Supports both streaming and non-streaming responses from Ollama or similar clients.
"""

from typing import Any, Dict, Union, Generator, Optional


class ResponseParser:
    """Utility class to normalize and extract responses from different LLM formats."""

    def __init__(self, stream: bool = False):
        self.stream = stream

    def parse(self, response: Union[Dict[str, Any], str, Generator]) -> str:
        """
        Parse the LLM response (streaming or non-streaming) into clean text.

        Args:
            response: Dict, string, or generator from Ollama/OpenAI-like client.

        Returns:
            Parsed and concatenated string output.
        """
        if self.stream:
            return self._parse_stream(response)
        return self._parse_static(response)

    def _parse_static(self, response: Union[Dict[str, Any], str]) -> str:
        """Parse non-streaming (synchronous) response."""
        if isinstance(response, str):
            return response.strip()

        if isinstance(response, dict):
            # Handle Ollama/OpenAI-like JSON responses
            text = (
                response.get("message", {}).get("content")
                or response.get("output_text")
                or response.get("response")
                or ""
            )
            return text.strip()

        return ""

    def _parse_stream(self, response: Generator) -> str:
        """Parse streaming (generator-based) response into a single string."""
        output = []
        for chunk in response:
            if isinstance(chunk, dict):
                content = (
                    chunk.get("message", {}).get("content")
                    or chunk.get("output_text")
                    or chunk.get("response")
                    or ""
                )
                output.append(content)
            elif isinstance(chunk, str):
                output.append(chunk)
        return "".join(output).strip()

    @staticmethod
    def extract_metadata(response: Union[Dict[str, Any], None]) -> Optional[Dict[str, Any]]:
        """
        Extract useful metadata (e.g., tokens, model info) from response.
        Non-fatal if fields don't exist.

        Returns:
            Dictionary containing token usage, model, latency, etc.
        """
        if not isinstance(response, dict):
            return None

        meta = {}
        for key in ("model", "created_at", "latency", "tokens", "usage", "elapsed"):
            if key in response:
                meta[key] = response[key]
        return meta or None




# from response_parser import ResponseParser

# # Example static response
# response = {
#     "message": {"content": "Hello, how can I help you today?"},
#     "model": "llama3",
#     "tokens": {"prompt": 12, "completion": 9}
# }

# parser = ResponseParser(stream=False)
# print(parser.parse(response))         # -> "Hello, how can I help you today?"
# print(parser.extract_metadata(response))  # -> {'model': 'llama3', 'tokens': {...}}

# # Example streaming response
# def fake_stream():
#     yield {"message": {"content": "Hello"}}
#     yield {"message": {"content": ", world!"}}

# stream_parser = ResponseParser(stream=True)
# print(stream_parser.parse(fake_stream()))  # -> "Hello, world!"