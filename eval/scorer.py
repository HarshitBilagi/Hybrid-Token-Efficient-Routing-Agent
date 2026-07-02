import sys
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bool(val: str) -> bool:
    return str(val).strip().upper() == "TRUE"


def score(results: list[dict]) -> dict:
    total = len(results)
    categories = defaultdict(lambda: {"total": 0, "correct": 0, "local": 0, "remote": 0})
    routes = defaultdict(lambda: {"total": 0, "correct": 0, "completion_tokens": 0, "latency_ms": 0})
    fallback_count = 0
    parse_failed_count = 0
    errors = 0

    always_remote_tokens = 0  # baseline: what remote would have cost for every query
    actual_remote_tokens = 0  # what we actually spent on remote

    for row in results:
        cat = row.get("category", "unknown")
        route = row.get("route_taken", "unknown")
        correct = parse_bool(row.get("correct", "false"))
        completion_tokens = int(row.get("completion_tokens") or 0)
        prompt_tokens = int(row.get("prompt_tokens") or 0)
        latency = int(row.get("latency_ms") or 0)
        fallback = parse_bool(row.get("fallback_triggered", "false"))
        error = row.get("error") or ""
        judge_failed = parse_bool(row.get("judge_parse_failed", "false"))

        categories[cat]["total"] += 1
        categories[cat]["correct"] += int(correct)
        categories[cat][route] = categories[cat].get(route, 0) + 1

        routes[route]["total"] += 1
        routes[route]["correct"] += int(correct)
        routes[route]["completion_tokens"] += completion_tokens
        routes[route]["latency_ms"] += latency

        if fallback:
            fallback_count += 1
        if judge_failed:
            parse_failed_count += 1
        if error:
            errors += 1

        # token economics
        # always-remote baseline: assume same completion tokens at remote cost
        always_remote_tokens += completion_tokens + prompt_tokens
        if route == "remote":
            actual_remote_tokens += completion_tokens + prompt_tokens
        # local costs 0 remote tokens

    total_correct = sum(r["correct"] for r in routes.values())
    accuracy = total_correct / total if total else 0

    token_savings = always_remote_tokens - actual_remote_tokens
    token_saving_pct = (token_savings / always_remote_tokens * 100) if always_remote_tokens else 0

    local_total = routes["local"]["total"]
    remote_total = routes["remote"]["total"]
    local_accuracy = (routes["local"]["correct"] / local_total) if local_total else None
    remote_accuracy = (routes["remote"]["correct"] / remote_total) if remote_total else None
    avg_local_latency = (routes["local"]["latency_ms"] / local_total) if local_total else None
    avg_remote_latency = (routes["remote"]["latency_ms"] / remote_total) if remote_total else None

    return {
        "summary": {
            "total_queries": total,
            "total_correct": total_correct,
            "accuracy": round(accuracy * 100, 1),
            "queries_routed_local": local_total,
            "queries_routed_remote": remote_total,
            "local_pct": round(local_total / total * 100, 1) if total else 0,
            "fallback_triggered": fallback_count,
            "judge_parse_failures": parse_failed_count,
            "errors": errors,
        },
        "token_economics": {
            "always_remote_baseline_tokens": always_remote_tokens,
            "actual_tokens_used": actual_remote_tokens,
            "tokens_saved": token_savings,
            "saving_pct": round(token_saving_pct, 1),
        },
        "accuracy_by_route": {
            "local": {
                "queries": local_total,
                "correct": routes["local"]["correct"],
                "accuracy": round(local_accuracy * 100, 1) if local_accuracy is not None else "n/a",
                "avg_latency_ms": round(avg_local_latency) if avg_local_latency else "n/a",
                "total_completion_tokens": routes["local"]["completion_tokens"],
            },
            "remote": {
                "queries": remote_total,
                "correct": routes["remote"]["correct"],
                "accuracy": round(remote_accuracy * 100, 1) if remote_accuracy is not None else "n/a",
                "avg_latency_ms": round(avg_remote_latency) if avg_remote_latency else "n/a",
                "total_completion_tokens": routes["remote"]["completion_tokens"],
            },
        },
        "accuracy_by_category": {
            cat: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for cat, v in categories.items()
        },
    }


def print_report(label: str, s: dict):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    sm = s["summary"]
    print(f"\n── Overall ──")
    print(f"  Queries        : {sm['total_queries']}")
    print(f"  Correct        : {sm['total_correct']} / {sm['total_queries']}  ({s['summary']['accuracy']}%)")
    print(f"  Routed local   : {sm['queries_routed_local']} ({sm['local_pct']}%)")
    print(f"  Routed remote  : {sm['queries_routed_remote']}")
    print(f"  Fallbacks      : {sm['fallback_triggered']}")
    print(f"  Judge failures : {sm['judge_parse_failures']}")

    te = s["token_economics"]
    print(f"\n── Token Economics ──")
    print(f"  Always-remote baseline : {te['always_remote_baseline_tokens']} tokens")
    print(f"  Actual tokens used     : {te['actual_tokens_used']} tokens")
    print(f"  Tokens saved           : {te['tokens_saved']}  ({te['saving_pct']}%)")

    print(f"\n── Accuracy by Route ──")
    for route, v in s["accuracy_by_route"].items():
        print(f"  {route:8s}: {v['correct']}/{v['queries']} correct "
              f"({v['accuracy']}%) | avg latency {v['avg_latency_ms']}ms "
              f"| {v['total_completion_tokens']} completion tokens")

    print(f"\n── Accuracy by Category ──")
    for cat, v in s["accuracy_by_category"].items():
        bar = "█" * v["correct"] + "░" * (v["total"] - v["correct"])
        print(f"  {cat:20s}: {v['correct']}/{v['total']} [{bar}]  {v['accuracy']}%")


def compare_report(label_a: str, s_a: dict, label_b: str, s_b: dict):
    print(f"\n{'=' * 60}")
    print(f"  Comparison: {label_a}  vs  {label_b}")
    print(f"{'=' * 60}")

    metrics = [
        ("Accuracy", f"{s_a['summary']['accuracy']}%", f"{s_b['summary']['accuracy']}%"),
        ("Local route %", f"{s_a['summary']['local_pct']}%", f"{s_b['summary']['local_pct']}%"),
        ("Token savings", f"{s_a['token_economics']['saving_pct']}%", f"{s_b['token_economics']['saving_pct']}%"),
        ("Fallbacks", s_a['summary']['fallback_triggered'], s_b['summary']['fallback_triggered']),
        ("Judge failures", s_a['summary']['judge_parse_failures'], s_b['summary']['judge_parse_failures']),
    ]

    print(f"\n  {'Metric':<22} {label_a:<20} {label_b}")
    print(f"  {'-'*60}")
    for metric, va, vb in metrics:
        print(f"  {metric:<22} {str(va):<20} {vb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score routing agent benchmark results.")
    parser.add_argument("--results", nargs="+", required=True, help="CSV result file(s)")
    parser.add_argument("--labels", nargs="+", help="Labels for each result file")
    parser.add_argument("--output-json", help="Optional path to write scores as JSON")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.results]
    all_scores = {}

    for path, label in zip(args.results, labels):
        print(f"Loading {path}...")
        results = load_csv(path)
        s = score(results)
        all_scores[label] = s
        print_report(label, s)

    if len(args.results) == 2:
        keys = list(all_scores.keys())
        compare_report(keys[0], all_scores[keys[0]], keys[1], all_scores[keys[1]])

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=2)
        print(f"\nScores written to {args.output_json}")