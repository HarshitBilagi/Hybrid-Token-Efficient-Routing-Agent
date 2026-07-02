import time
from enum import Enum


class ClassifierType(str, Enum):
    RULE_BASED = "rule_based"
    LLM_JUDGED = "llm_judged"


class RoutingAgent:
    """
    Combines a classifier with local + remote inference clients.
    This is the core orchestration layer — accepts a query, decides
    a route, dispatches, and returns a full response envelope.

    Includes runtime fallback: if local inference fails or raises,
    automatically retries on remote rather than erroring out to the caller.
    """

    def __init__(
        self,
        local_pipeline,
        remote_client,
        rule_based_classifier,
        llm_judged_classifier=None,
        default_classifier: ClassifierType = ClassifierType.RULE_BASED,
    ):
        self.local_pipeline = local_pipeline
        self.remote_client = remote_client
        self.rule_based_classifier = rule_based_classifier
        self.llm_judged_classifier = llm_judged_classifier
        self.default_classifier = default_classifier

    def route(
        self,
        query: str,
        context: list[dict] = None,
        classifier: ClassifierType = None,
        prefer_local: bool = False,
        max_latency_ms: int = None,
    ) -> dict:
        """
        Full routing pipeline: classify -> dispatch -> respond.

        Returns a response envelope:
            {
                response, route_taken, classifier_used, classifier_signals,
                tokens, latency_ms, model_used, fallback_triggered
            }
        """
        classifier_type = classifier or self.default_classifier
        context = context or []

        classification, classify_latency_ms = self._classify(query, classifier_type)
        route_recommendation = classification["route_recommendation"]

        # constraint override: hard latency ceiling forces remote,
        # since local NPU throughput (~10 tok/s) can't guarantee it
        if max_latency_ms is not None and route_recommendation == "local":
            estimated_local_ms = classification.get("predicted_output_tokens", 80) * 100  # ~10 tok/s
            if estimated_local_ms > max_latency_ms:
                route_recommendation = "remote"

        # manual override: caller can force local regardless of classifier
        if prefer_local:
            route_recommendation = "local"

        fallback_triggered = False

        if route_recommendation == "local":
            try:
                result = self._generate_local(query, context, classification)
            except Exception as e:
                fallback_triggered = True
                fallback_reason = f"{type(e).__name__}: {e}"
                print(f"[Router] Local inference failed, falling back to remote: {fallback_reason}")
                result = self._generate_remote(query, context)
        else:
            fallback_reason = None
            result = self._generate_remote(query, context)

        return {
            "response": result["response"],
            "route_taken": "local" if (route_recommendation == "local" and not fallback_triggered) else "remote",
            "classifier_used": classifier_type.value,
            "classifier_signals": classification.get("signals", {}),
            "classifier_latency_ms": classify_latency_ms,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason if fallback_triggered else None,
            "tokens": {
                "completion": result.get("tokens_generated", 0),
                "prompt": result.get("prompt_tokens"),
            },
            "latency_ms": result["latency_ms"],
            "model_used": result["model_used"],
        }

    def _classify(self, query: str, classifier_type: ClassifierType) -> tuple[dict, int]:
        t0 = time.perf_counter()

        if classifier_type == ClassifierType.LLM_JUDGED:
            if self.llm_judged_classifier is None:
                raise ValueError("LLM-judged classifier not configured on this router.")
            result = self.llm_judged_classifier.classify(query)
        else:
            result = self.rule_based_classifier.classify(query)

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return result, elapsed_ms

    def _generate_local(self, query: str, context: list[dict], classification: dict) -> dict:
        predicted_tokens = classification.get("predicted_output_tokens", 80)
        # cap generation slightly above prediction, never runaway
        max_tokens = min(predicted_tokens + 40, 256)
        return self.local_pipeline.generate(query, max_new_tokens=max_tokens, context=context)

    def _generate_remote(self, query: str, context: list[dict]) -> dict:
        return self.remote_client.generate(query, context=context, max_new_tokens=512)