import time
import openvino_genai as ov_genai
from pathlib import Path


class PhiNPUPipeline:
    """
    Wraps OpenVINO GenAI LLMPipeline for Phi-3.5-mini on the Intel AI Boost NPU.
    Exposes a simple generate() interface consumed by the router.
    """

    DEFAULT_MODEL_PATH = "./models/phi35-mini-int4-cw-ov"

    def __init__(self, model_path: str = None, device: str = "NPU"):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = device
        self._pipe = None

    def load(self):
        """
        Load and compile the model onto the target device.
        First call compiles the NPU blob (~30s). Subsequent calls use cache (~5s).
        """
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"[PhiPipeline] Loading model on {self.device}...")
        t0 = time.perf_counter()
        self._pipe = ov_genai.LLMPipeline(self.model_path, device=self.device)
        elapsed = time.perf_counter() - t0
        print(f"[PhiPipeline] Ready in {elapsed:.1f}s")

    def generate(
        self,
        query: str,
        max_new_tokens: int = 256,
        context: list[dict] = None,
    ) -> dict:
        """
        Run inference. Returns a result dict matching the response envelope.

        Args:
            query: raw user query string
            max_new_tokens: cap on generated tokens (keep low for latency)
            context: optional list of {role, content} prior turns

        Returns:
            {response, tokens_generated, latency_ms, model_used}
        """
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        prompt = self._build_prompt(query, context or [])

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens

        token_count = 0

        def _count(token: str):
            nonlocal token_count
            token_count += 1

        t0 = time.perf_counter()
        result = self._pipe.generate(prompt, config, streamer=_count)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Handle both DecodedResults (with .texts attribute) and standard string outputs
        response_text = result.texts[0] if hasattr(result, "texts") else result

        return {
            "response": response_text.strip(),
            "tokens_generated": token_count,
            "latency_ms": round(latency_ms),
            "model_used": "Phi-3.5-mini-instruct-int4-gq",
        }

    def _build_prompt(self, query: str, context: list[dict]) -> str:
        """
        Formats query + context into Phi-3.5 chat template.
        """
        parts = []
        for turn in context:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"<|{role}|>\n{content}<|end|>")
        parts.append(f"<|user|>\n{query}<|end|>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def is_loaded(self) -> bool:
        return self._pipe is not None

        