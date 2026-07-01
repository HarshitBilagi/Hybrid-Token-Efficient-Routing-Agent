import sys
sys.path.insert(0, ".")

from src.local_inference.phi_pipeline import PhiNPUPipeline
from src.remote_inference.remote_client import get_remote_client
from src.classifier.rule_based import RuleBasedClassifier
from src.classifier.llm_judged import LLMJudgedClassifier
from src.router.router import RoutingAgent, ClassifierType

pipeline = PhiNPUPipeline()
pipeline.load()

remote = get_remote_client()
rule_clf = RuleBasedClassifier()
llm_clf = LLMJudgedClassifier(pipeline)

agent = RoutingAgent(
    local_pipeline=pipeline,
    remote_client=remote,
    rule_based_classifier=rule_clf,
    llm_judged_classifier=llm_clf,
    default_classifier=ClassifierType.RULE_BASED,
)

test_queries = [
    "What is the capital of France?",
    "Who is the current president of the United States?",
    "Write a Python function that reverses a string.",
]

for q in test_queries:
    result = agent.route(q)
    print(f"Query: {q}")
    print(f"  route_taken: {result['route_taken']} | classifier: {result['classifier_used']}")
    print(f"  latency: {result['latency_ms']}ms | tokens: {result['tokens']}")
    print(f"  fallback_triggered: {result['fallback_triggered']}")
    print(f"  response: {result['response']}\n")
    print("=" * 60)
    
print("Testing LLM_JUDGED classifier path")
print("=" * 60)

result = agent.route(
    "What is the capital of France?",
    classifier=ClassifierType.LLM_JUDGED,
)
print(f"route_taken: {result['route_taken']} | classifier: {result['classifier_used']}")
print(f"classifier_latency_ms: {result['classifier_latency_ms']}")
print(f"signals: {result['classifier_signals']}")
print(f"response: {result['response']}\n")

print("=" * 60)
print("Testing max_latency_ms constraint (forces remote even for simple query)")
print("=" * 60)

result = agent.route(
    "What is the capital of France?",
    max_latency_ms=2000,  # 2s ceiling — local NPU can't hit this
)
print(f"route_taken: {result['route_taken']} | latency: {result['latency_ms']}ms")
print(f"response: {result['response']}\n")

print("=" * 60)
print("Testing prefer_local override (forces local even for code query)")
print("=" * 60)

result = agent.route(
    "Write a Python function that reverses a string.",
    prefer_local=True,
)
print(f"route_taken: {result['route_taken']} | latency: {result['latency_ms']}ms")
print(f"response: {result['response'][:150]}...\n")