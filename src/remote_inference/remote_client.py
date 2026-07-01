import os
import time
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RemoteInferenceClient(ABC):
    """
    Provider-agnostic interface for remote inference.
    Swap providers (Groq -> Fireworks) by changing config, not router logic.
    """

    @abstractmethod
    def generate(self, query: str, context: list[dict] = None, max_new_tokens: int = 512) -> dict:
        ...


class OpenAICompatibleClient(RemoteInferenceClient):
    """
    Works for any OpenAI-compatible provider: Groq, Fireworks, Together, etc.
    Just change base_url, api_key, and model.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, query: str, context: list[dict] = None, max_new_tokens: int = 512) -> dict:
        messages = list(context or [])
        messages.append({"role": "user", "content": query})

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0].message.content
        usage = response.usage

        return {
            "response": choice.strip(),
            "tokens_generated": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
            "latency_ms": round(latency_ms),
            "model_used": self.model,
        }


def get_remote_client() -> RemoteInferenceClient:
    """
    Factory function. This is the ONLY place provider selection happens.
    Swap 'groq' -> 'fireworks' here when hackathon starts, nothing else changes.
    """
    provider = os.getenv("REMOTE_PROVIDER", "groq")

    if provider == "groq":
        return OpenAICompatibleClient(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
        )
    elif provider == "fireworks":
        return OpenAICompatibleClient(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")