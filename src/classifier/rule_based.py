# KNOWN LIMITATION: this classifier has no concept of "requires current knowledge."
# Queries like "who is the current president" pattern-match as simple and route
# local, but a static local model may hallucinate outdated answers. This is a
# primary motivator for the LLM-judged classifier (self-assessed confidence
# accounts for the model's own knowledge cutoff).

import re


class RuleBasedClassifier:
    """
    Heuristic query classifier. No model inference — pure pattern matching.
    Serves as baseline for comparison against the LLM-judged classifier.

    Outputs a routing recommendation based on predicted output length
    and query complexity signals.
    """

    CODE_KEYWORDS = {
        "write a function", "implement", "debug", "fix this code",
        "python", "javascript", "algorithm", "class", "def ", "code for",
        "regex", "sql query", "script",
    }

    REASONING_KEYWORDS = {
        "prove", "derive", "calculate", "solve", "explain why", "explain how",
        "compare", "analyze", "evaluate", "what if", "trade-off", "tradeoff",
    }

    SIMPLE_PATTERNS = [
        r"^what is\b", r"^who is\b", r"^when (was|did|is)\b",
        r"^where is\b", r"^define\b", r"^how many\b",
    ]

    def __init__(self, short_output_threshold: int = 80):
        self.short_output_threshold = short_output_threshold

    def classify(self, query: str) -> dict:
        """
        Returns:
            {
                predicted_complexity: "simple" | "medium" | "complex",
                predicted_output_tokens: int (rough estimate),
                route_recommendation: "local" | "remote",
                signals: dict (for debugging/eval logging)
            }
        """
        q = query.strip().lower()
        word_count = len(q.split())
        question_marks = q.count("?")

        has_code = any(kw in q for kw in self.CODE_KEYWORDS)
        has_reasoning = any(kw in q for kw in self.REASONING_KEYWORDS)
        is_simple_pattern = any(re.match(p, q) for p in self.SIMPLE_PATTERNS)
        is_multi_part = question_marks > 1 or " and " in q

        # --- complexity scoring ---
        complexity_score = 0
        if has_code:
            complexity_score += 2
        if has_reasoning:
            complexity_score += 2
        if is_multi_part:
            complexity_score += 1
        if word_count > 20:
            complexity_score += 1
        if is_simple_pattern:
            complexity_score -= 2

        if complexity_score <= 0:
            complexity = "simple"
            predicted_tokens = 40
        elif complexity_score <= 2:
            complexity = "medium"
            predicted_tokens = 120
        else:
            complexity = "complex"
            predicted_tokens = 300

        # --- routing decision ---
        # local only for short, simple, non-code queries
        if (
            complexity == "simple"
            and predicted_tokens <= self.short_output_threshold
            and not has_code
        ):
            route = "local"
        else:
            route = "remote"

        return {
            "predicted_complexity": complexity,
            "predicted_output_tokens": predicted_tokens,
            "route_recommendation": route,
            "signals": {
                "word_count": word_count,
                "has_code": has_code,
                "has_reasoning": has_reasoning,
                "is_simple_pattern": is_simple_pattern,
                "is_multi_part": is_multi_part,
                "complexity_score": complexity_score,
            },
        }