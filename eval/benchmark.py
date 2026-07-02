import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, ".")

from src.local_inference.phi_pipeline import PhiNPUPipeline
from src.remote_inference.remote_client import get_remote_client
from src.classifier.rule_based import RuleBasedClassifier
from src.classifier.llm_judged import LLMJudgedClassifier
from src.router.router import RoutingAgent, ClassifierType


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(dataset_path: str, classifier: ClassifierType, output_prefix: str):
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"{len(dataset)} queries loaded.\n")

    print("Initializing pipeline and router...")
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
        default_classifier=classifier,
    )

    results = []

    for i, item in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] {item['id']}: {item['query'][:60]}...")

        t0 = time.perf_counter()
        try:
            route_result = agent.route(item["query"], classifier=classifier)
            error = None
        except Exception as e:
            route_result = {
                "response": None,
                "route_taken": None,
                "classifier_used": classifier.value,
                "classifier_signals": {},
                "classifier_latency_ms": None,
                "fallback_triggered": False,
                "fallback_reason": None,
                "tokens": {"completion": 0, "prompt": None},
                "latency_ms": 0,
                "model_used": None,
            }
            error = str(e)
        total_ms = round((time.perf_counter() - t0) * 1000)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "query": item["query"],
            "expected_answer": item["expected_answer"],
            "response": route_result["response"],
            "route_taken": route_result["route_taken"],
            "classifier_used": route_result["classifier_used"],
            "fallback_triggered": route_result["fallback_triggered"],
            "fallback_reason": route_result.get("fallback_reason"),
            "completion_tokens": route_result["tokens"]["completion"],
            "prompt_tokens": route_result["tokens"].get("prompt"),
            "latency_ms": route_result["latency_ms"],
            "total_wall_ms": total_ms,
            "model_used": route_result["model_used"],
            "error": error,
            "correct": None,  # fill in manually after reviewing response vs expected_answer
        })

    # write outputs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{output_prefix}_{timestamp}.json"
    csv_path = out_dir / f"{output_prefix}_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results written to:\n  {json_path}\n  {csv_path}")
    print(f"\nNext step: open the CSV, review 'response' vs 'expected_answer' for each row,")
    print(f"and fill in 'correct' as TRUE or FALSE.")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/datasets/benchmark_v1.json")
    parser.add_argument("--classifier", choices=["rule_based", "llm_judged"], default="rule_based")
    parser.add_argument("--output-prefix", default="benchmark")
    args = parser.parse_args()

    classifier_enum = ClassifierType(args.classifier)
    run_benchmark(args.dataset, classifier_enum, args.output_prefix)