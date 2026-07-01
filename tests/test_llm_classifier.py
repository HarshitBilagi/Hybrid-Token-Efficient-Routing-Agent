import sys
sys.path.insert(0, ".")

from src.local_inference.phi_pipeline import PhiNPUPipeline
from src.classifier.llm_judged import LLMJudgedClassifier

pipeline = PhiNPUPipeline()
pipeline.load()

clf = LLMJudgedClassifier(pipeline)

test_queries = [
    "What is the capital of France?",
    "Who is the current president of the United States?",
    "Write a Python function that reverses a string.",
    "What's the latest iPhone model?",
]

for q in test_queries:
    result = clf.classify(q)
    print(f"Query: {q}")
    print(f"  -> route: {result['route_recommendation']} | complexity: {result.get('predicted_complexity')}")
    print(f"  signals: {result.get('signals')}")
    print(f"  parse_failed: {result.get('judge_parse_failed')} | judge_latency: {result.get('judge_latency_ms')}ms\n")