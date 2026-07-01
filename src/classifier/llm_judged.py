import json
import re

from src.classifier.rule_based import RuleBasedClassifier


class LLMJudgedClassifier:
    """
    Uses the local Phi-3.5-mini pipeline to self-assess query difficulty
    and knowledge-currency requirements. Falls back to RuleBasedClassifier
    if the model's JSON output fails to parse OR fails validation.

    Note: INT4-quantized 3.8B models are unreliable JSON generators. This
    classifier treats every parsed field as untrusted input and validates
    types/ranges before use, rather than assuming well-formed output.
    """

    JUDGE_PROMPT_TEMPLATE = (
        "Assess the following query. Respond with ONLY a JSON object, "
        "no other text, no markdown formatting, no trailing commas.\n\n"
        "Query: \"{query}\"\n\n"
        "Return JSON with exactly these fields:\n"
        "{{\n"
        '  "confidence": <float between 0 and 1>,\n'
        '  "requires_current_knowledge": <true or false>,\n'
        '  "predicted_complexity": <"simple", "medium", or "complex">,\n'
        '  "predicted_output_tokens": <integer>\n'
        "}}"
    )

    def __init__(self, local_pipeline, fallback_classifier: RuleBasedClassifier = None):
        self.pipeline = local_pipeline
        self.fallback = fallback_classifier or RuleBasedClassifier()

    def classify(self, query: str, short_output_threshold: int = 80) -> dict:
        prompt = self.JUDGE_PROMPT_TEMPLATE.format(query=query)
        result = self.pipeline.generate(prompt, max_new_tokens=100)
        raw_output = result["response"]

        parsed = self._parse_and_validate(raw_output)

        # cheap secondary signal, always computed regardless of LLM parse success
        rule_signals = self.fallback.classify(query)["signals"]
        has_code = rule_signals.get("has_code", False)

        if parsed is None:
            fallback_result = self.fallback.classify(query)
            fallback_result["judge_parse_failed"] = True
            fallback_result["raw_judge_output"] = raw_output
            fallback_result["judge_latency_ms"] = result["latency_ms"]
            return fallback_result

        requires_current = parsed["requires_current_knowledge"]
        confidence = parsed["confidence"]
        complexity = parsed["predicted_complexity"]
        predicted_tokens = parsed["predicted_output_tokens"]

        if requires_current or confidence < 0.6 or has_code:
            route = "remote"
        elif complexity == "simple" and predicted_tokens <= short_output_threshold:
            route = "local"
        else:
            route = "remote"

        return {
            "predicted_complexity": complexity,
            "predicted_output_tokens": predicted_tokens,
            "route_recommendation": route,
            "signals": {
                "confidence": confidence,
                "requires_current_knowledge": requires_current,
                "has_code": has_code,
            },
            "judge_parse_failed": False,
            "judge_latency_ms": result["latency_ms"],
        }

    def _parse_and_validate(self, text: str) -> dict | None:
        """
        Extracts JSON, then validates and coerces every field.
        Returns None if extraction fails OR any field fails validation —
        both cases trigger the rule-based fallback.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        # strip trailing commas before closing braces/brackets — common
        # quantization artifact ("0.98,}" instead of "0.98}")
        cleaned = re.sub(r",\s*([}\]])", r"\1", match.group())

        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        try:
            confidence = float(str(raw["confidence"]).rstrip(","))
            if not (0.0 <= confidence <= 1.0):
                return None

            requires_current = raw["requires_current_knowledge"]
            if isinstance(requires_current, str):
                requires_current = requires_current.strip().lower() == "true"
            if not isinstance(requires_current, bool):
                return None

            complexity = str(raw["predicted_complexity"]).strip().lower()
            if complexity not in {"simple", "medium", "complex"}:
                return None

            predicted_tokens = int(float(str(raw["predicted_output_tokens"]).rstrip(",")))
            if not (0 < predicted_tokens <= 2000):
                return None

        except (KeyError, ValueError, TypeError):
            return None

        return {
            "confidence": confidence,
            "requires_current_knowledge": requires_current,
            "predicted_complexity": complexity,
            "predicted_output_tokens": predicted_tokens,
        }