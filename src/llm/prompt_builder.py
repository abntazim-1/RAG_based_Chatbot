# src/llm/prompt_builder.py
"""
PromptBuilder
-------------
A modular utility to construct high-quality prompts for RAG-based chatbots.

Responsibilities:
- Format retrieved context and user query into a clean, model-ready prompt.
- Support different prompt templates (e.g., QA, summarization, contextual chat).
- Allow easy extension for few-shot examples or instruction-based prompting.

Example:
    pb = PromptBuilder()
    prompt = pb.build_qa_prompt(context="RAG improves factual grounding.", query="What is RAG?")
    print(prompt)
"""

from __future__ import annotations
from typing import List, Dict, Optional


class PromptBuilder:
    """A minimal, extensible class for building structured LLM prompts."""

    def __init__(self, system_instruction: Optional[str] = None) -> None:
        self.system_instruction = (
            system_instruction
            or "You are a helpful AI assistant. Use the given context to answer accurately."
        )

    def build_qa_prompt(self, context: str, query: str) -> str:
        """Constructs a prompt for retrieval-augmented QA."""
        prompt = (
            f"{self.system_instruction}\n\n"
            f"Use the following context from the documents to answer the question. "
            f"If the context doesn't contain enough information, say so.\n\n"
            f"Context from documents:\n{context.strip()}\n\n"
            f"Question: {query.strip()}\n\n"
            f"Answer based on the context above:"
        )
        return prompt

    def build_summary_prompt(self, context: str, max_length: int = 150) -> str:
        """Constructs a prompt for summarization tasks."""
        prompt = (
            f"{self.system_instruction}\n\n"
            f"Summarize the following text in under {max_length} words:\n\n"
            f"{context.strip()}\n\nSummary:"
        )
        return prompt

    def build_chat_prompt(self, history: List[Dict[str, str]], user_message: str) -> str:
        """Constructs a conversational prompt from chat history and new user input."""
        history_text = "\n".join(
            [f"{h['role']}: {h['content'].strip()}" for h in history if 'role' in h and 'content' in h]
        )
        prompt = (
            f"{self.system_instruction}\n\n"
            f"{history_text}\nuser: {user_message.strip()}\nassistant:"
        )
        return prompt

    def build_custom_prompt(self, template: str, variables: Dict[str, str]) -> str:
        """Supports dynamic prompt templates using placeholders."""
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing placeholder in template: {e}")


if __name__ == "__main__":
    # Example usage test
    builder = PromptBuilder()
    context_text = "Retrieval-Augmented Generation (RAG) combines LLMs with external knowledge sources."
    query_text = "Explain how RAG works in one sentence."

    prompt = builder.build_qa_prompt(context_text, query_text)
    print("Generated Prompt:\n", prompt)
