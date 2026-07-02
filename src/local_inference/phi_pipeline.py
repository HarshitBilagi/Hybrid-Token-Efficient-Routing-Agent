import time
from pathlib import Path

import openvino_genai as ov_genai


class PhiNPUPipeline:
    """
    Generic OpenVINO GenAI wrapper for local NPU inference.
    Model-agnostic: swap models by changing model_path and chat_template.
    """

    DEFAULT_MODEL_PATH = "./models/phi35-mini-int4-cw-ov"
    DEFAULT_CHAT_TEMPLATE = "<|user|>\n{content}<|end|>\n<|assistant|>"  # Phi-3.5 format

    def __init__(self, model_path: str = None, device: str = "NPU", chat_template: str = None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = device
        self.chat_template = chat_template or self.DEFAULT_CHAT_TEMPLATE
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
        Formats query + context using the configured chat template.
        Model-agnostic: change chat_template to support other model families.
        """
        parts = []
        for turn in context:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(self.chat_template.format(content=f"({role}) {content}"))
        parts.append(self.chat_template.format(content=query))
        return "\n".join(parts)

    def is_loaded(self) -> bool:
        return self._pipe is not None