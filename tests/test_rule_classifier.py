import sys
sys.path.insert(0, ".")

from src.classifier.rule_based import RuleBasedClassifier

clf = RuleBasedClassifier()

test_queries = [
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "Explain the difference between TCP and UDP in one paragraph.",
    "Who is the current president of the United States?",
    "Prove that the square root of 2 is irrational and explain why this matters.",
    "How many continents are there?",
]

for q in test_queries:
    result = clf.classify(q)
    print(f"Query: {q}")
    print(f"  -> {result['predicted_complexity']} | route: {result['route_recommendation']} | ~{result['predicted_output_tokens']} tok")
    print(f"  signals: {result['signals']}\n")