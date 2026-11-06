import os
from typing import Protocol

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class LLMProvider(Protocol):
    """Interface that every LLM provider must implement."""

    def generate(self, prompt: str) -> str:
        ...


class MockLLM:
    """
    Local LLM implementation used for testing.
    Does not require an API call.
    """

    def generate(self, prompt: str) -> str:
        return "This is a test response from the mock LLM."


class GroqLLM:
    """
    Groq-based LLM implementation.
    """

    def __init__(
        self,
        model: str | None = None,
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured in the environment."
            )

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        self.client = Groq(api_key=api_key)


    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support assistant. "
                        "Answer using the provided verified context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # Track actual API token usage
        usage = response.usage

        if usage:
            print("\n========== GROQ TOKEN USAGE ==========")
            print(f"Model: {self.model}")
            print(f"Input tokens: {usage.prompt_tokens}")
            print(f"Output tokens: {usage.completion_tokens}")
            print(f"Total tokens: {usage.total_tokens}")
            print("======================================\n")

        return response.choices[0].message.content or ""
