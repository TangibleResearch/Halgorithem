from collections import Counter, defaultdict, deque

from Halgorithem import Halgorithm


PASS_THRESHOLD = 0.90

TRUTH_DOCS = [
    {
        "file_id": 1,
        "file_path": "sources/basic.txt",
        "text": (
            "BASIC is a programming language created in 1964 by John Kemeny and "
            "Thomas Kurtz at Dartmouth College. BASIC was designed for students. "
            "BASIC was used on time-sharing systems. Early BASIC used interpreters. "
            "John Kemeny created BASIC. Thomas Kurtz created BASIC."
        ),
    },
    {
        "file_id": 2,
        "file_path": "sources/science.txt",
        "text": (
            "Alexander Fleming discovered penicillin in 1928. Marie Curie discovered "
            "radium in 1898. The Mars sample container has a mass of 10 kilograms. "
            "Mount Everest is 8849 meters tall. The Python language was created by "
            "Guido van Rossum in 1991. Guido van Rossum created Python."
        ),
    },
    {
        "file_id": 3,
        "file_path": "sources/table.txt",
        "text": (
            "Table: city, country, population. Paris, France, 2148000. "
            "Tokyo, Japan, 14000000. Lima, Peru, 9752000. "
            "Product A price 19 USD. Product B price 25 USD."
        ),
    },
    {
        "file_id": 4,
        "file_path": "sources/current.txt",
        "text": (
            "As of 2024, Project Helios has status active. The latest release of "
            "Project Helios is version 3.2. The current maintainer is Dana Lee."
        ),
    },
    {
        "file_id": 5,
        "file_path": "sources/report-alpha.txt",
        "text": "Report Alpha says Northwind revenue is 10 USD and profit is 4 USD.",
    },
    {
        "file_id": 6,
        "file_path": "sources/report-beta.txt",
        "text": "Report Beta says Northwind revenue is 12 USD and profit is 5 USD.",
    },
]


def numbered(category, claims, expected, warning=False):
    return [
        {
            "id": f"{category}-{i:03d}",
            "category": category,
            "claim": claim,
            "expected": expected,
            "warning": warning,
        }
        for i, claim in enumerate(claims, 1)
    ]


def cycle_to(items, count):
    return [items[i % len(items)] for i in range(count)]


def build_test_cases():
    supported = cycle_to([
        "BASIC was created in 1964.",
        "BASIC was created by John Kemeny.",
        "BASIC was created by Thomas Kurtz.",
        "BASIC was designed for students.",
        "BASIC was used on time-sharing systems.",
        "Early BASIC used interpreters.",
        "Alexander Fleming discovered penicillin in 1928.",
        "Marie Curie discovered radium in 1898.",
        "Mount Everest is 8849 meters tall.",
        "Python was created by Guido van Rossum in 1991.",
    ], 50)

    paraphrases = cycle_to([
        "The BASIC programming language was created in 1964.",
        "The BASIC language was designed for students.",
        "BASIC was used on time-sharing systems.",
        "Python language was created by Guido van Rossum.",
        "Penicillin was discovered by Alexander Fleming.",
        "Radium was discovered by Marie Curie.",
        "Mount Everest is 8849 meters tall.",
        "Product A has price 19 USD.",
        "Product B has price 25 USD.",
        "Tokyo, Japan, has population 14000000.",
    ], 50)

    weak = cycle_to([
        "BASIC helped beginners learn programming.",
        "BASIC made programming easier for students.",
        "BASIC was important for programming education.",
        "BASIC was influential in early computing.",
        "Time-sharing helped make BASIC useful for students.",
        "Python helped popularize readable programming.",
        "Project Helios status helped teams track releases.",
        "The table made city facts easier to compare.",
        "Interpreters helped BASIC feel interactive.",
        "BASIC was significant for beginner programmers.",
    ], 50)

    hallucinations = cycle_to([
        "BASIC was created by NASA.",
        "BASIC was invented in Germany.",
        "Python was created by NASA.",
        "Penicillin was discovered by Ada Lovelace.",
        "Mount Everest is located in Peru.",
        "Project Helios is maintained by Grace Hopper.",
        "Product C has price 31 USD.",
        "Lima is located in Japan.",
        "Northwind revenue is reported by Report Gamma.",
        "The Mars sample container was built by Microsoft.",
    ], 50)

    date_mismatch = cycle_to([
        "BASIC was created in 1972.",
        "Alexander Fleming discovered penicillin in 1935.",
        "Marie Curie discovered radium in 1905.",
        "Python was created by Guido van Rossum in 1988.",
        "Project Helios latest release was version 3.2 in 2021.",
    ], 50)

    role_swaps = cycle_to([
        "Marie Curie discovered penicillin.",
        "Alexander Fleming discovered radium.",
        "John Kemeny created Python.",
        "Guido van Rossum created BASIC.",
        "Thomas Kurtz created Python.",
    ], 50)

    unit_errors = cycle_to([
        "The Mars sample container has a mass of 10 pounds.",
        "Mount Everest is 8849 miles tall.",
        "Product A price is 19 EUR.",
        "Product B price is 25 EUR.",
        "Report Alpha says Northwind revenue is 10 EUR.",
    ], 50)

    current_latest = cycle_to([
        "The current status of Project Helios is active.",
        "The latest release of Project Helios is version 3.2.",
        "The current maintainer is Dana Lee.",
        "As of today, Project Helios has status active.",
        "Now Project Helios is maintained by Dana Lee.",
    ], 50)

    table_like = cycle_to([
        "Paris is in France.",
        "Tokyo is in Japan.",
        "Lima is in Peru.",
        "Paris has population 2148000.",
        "Tokyo has population 14000000.",
        "Lima has population 9752000.",
        "Product A price is 19 USD.",
        "Product B price is 25 USD.",
        "Paris, France, has population 2148000.",
        "Product B is priced at 25 USD.",
    ], 50)

    disagreement = cycle_to([
        "Report Alpha says Northwind revenue is 12 USD.",
        "Report Beta says Northwind revenue is 10 USD.",
        "Report Alpha says Northwind profit is 5 USD.",
        "Report Beta says Northwind profit is 4 USD.",
        "Report Alpha says Northwind revenue is 10 EUR.",
    ], 50)

    denials = cycle_to([
        "BASIC was not created by NASA.",
        "Python was not created by NASA.",
        "Penicillin was not discovered by Ada Lovelace.",
        "Project Helios is not maintained by Grace Hopper.",
        "Product C is not priced at 31 USD.",
        "Report Gamma does not report Northwind revenue.",
        "BASIC was never invented in Germany.",
        "Lima is not located in Japan.",
        "The Mars sample container was not built by Microsoft.",
        "Mount Everest is not located in Peru.",
    ], 50)

    missing_source = cycle_to([
        "The Moonbase Atlas budget is 44 USD.",
        "The Zeta protocol was approved in 2020.",
        "Orion City has population 12345.",
        "Quantum Fruit version 9.1 is current.",
        "The Argo tablet weighs 7 kilograms.",
    ], 50)

    return (
        numbered("supported", supported, "SUPPORTED")
        + numbered("paraphrase", paraphrases, "SUPPORTED")
        + numbered("weak_support", weak, "WEAK_SUPPORT")
        + numbered("hallucination", hallucinations, "HALLUCINATION")
        + numbered("date_mismatch", date_mismatch, "CONTRADICTION")
        + numbered("entity_role_swap", role_swaps, "CONTRADICTION")
        + numbered("unit_error", unit_errors, "CONTRADICTION")
        + numbered("current_latest", current_latest, "SUPPORTED", warning=True)
        + numbered("table_like", table_like, "SUPPORTED")
        + numbered("multi_source_disagreement", disagreement, "CONTRADICTION")
        + numbered("denial", denials, "UNVERIFIABLE_DENIAL")
        + numbered("missing_source", missing_source, "HALLUCINATION")
    )


def run_benchmark():
    algo = Halgorithm(sentences_per_chunk=2, sentence_overlap=1)
    test_cases = build_test_cases()
    expected_by_claim = defaultdict(deque)
    for case in test_cases:
        expected_by_claim[algo.clean_text(case["claim"])].append(case)

    results = algo.compare_to_docs(
        truth_docs=TRUTH_DOCS,
        ai_output="\n".join(case["claim"] for case in test_cases),
        threshold=0.30,
    )

    correct = 0
    warning_correct = 0
    warning_total = sum(1 for case in test_cases if case["warning"])
    failures = []
    expected_counts = Counter(case["expected"] for case in test_cases)
    predicted_counts = Counter()
    category_totals = Counter(case["category"] for case in test_cases)
    category_correct = Counter()
    confusion = defaultdict(Counter)

    print("\nBenchmark Report")
    print("=" * 80)
    print(f"Test cases: {len(test_cases)}")
    print(f"Verifier results: {len(results)}")

    if len(results) != len(test_cases):
        print("\nWARNING: result count differs from test-case count.")

    for result in results:
        claim = result["claim"]
        matching = expected_by_claim.get(claim)
        if not matching:
            failures.append({
                "id": "unknown",
                "category": "unknown",
                "claim": claim,
                "expected": "NO_MATCHING_TEST_CASE",
                "predicted": result["status"],
                "score": result.get("score", 0.0),
                "reason": result.get("reason", ""),
            })
            continue

        case = matching.popleft()
        expected = case["expected"]
        predicted = result["status"]
        predicted_counts[predicted] += 1
        confusion[expected][predicted] += 1

        has_warning = bool(result.get("warning"))
        warning_ok = not case["warning"] or has_warning
        if case["warning"] and has_warning:
            warning_correct += 1

        if predicted == expected and warning_ok:
            correct += 1
            category_correct[case["category"]] += 1
        else:
            failures.append({
                "id": case["id"],
                "category": case["category"],
                "claim": claim,
                "expected": expected,
                "predicted": predicted,
                "score": result.get("score", 0.0),
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason", ""),
                "warning": result.get("warning"),
                "unsupported_terms": result.get("unsupported_terms", []),
                "chunk_text": result.get("chunk_text", ""),
            })

    for remaining in expected_by_claim.values():
        for case in remaining:
            confusion[case["expected"]]["NO_RESULT"] += 1
            failures.append({
                "id": case["id"],
                "category": case["category"],
                "claim": algo.clean_text(case["claim"]),
                "expected": case["expected"],
                "predicted": "NO_RESULT",
                "score": 0.0,
                "confidence": 0.0,
                "reason": "Claim was not returned by the verifier.",
            })

    accuracy = correct / len(test_cases)
    print("\nExpected labels")
    for status, count in sorted(expected_counts.items()):
        print(f"  {status:22} {count}")

    print("\nPredicted labels")
    for status, count in sorted(predicted_counts.items()):
        print(f"  {status:22} {count}")

    print("\nAccuracy by category")
    for category, total in sorted(category_totals.items()):
        pct = category_correct[category] / total
        print(f"  {category:28} {category_correct[category]:3d}/{total:<3d} {pct * 100:6.2f}%")

    print("\nConfusion matrix")
    labels = sorted(set(expected_counts) | set(predicted_counts) | {"NO_RESULT"})
    print("  expected \\ predicted".ljust(30) + " ".join(label[:8].rjust(8) for label in labels))
    for expected in labels:
        row = [confusion[expected][predicted] for predicted in labels]
        print(f"  {expected[:26].ljust(28)}" + " ".join(str(value).rjust(8) for value in row))

    print(f"\nTemporal warning checks: {warning_correct}/{warning_total}")

    if failures:
        grouped = Counter((f["expected"], f["predicted"]) for f in failures)
        print("\nFailure summary")
        for (expected, predicted), count in sorted(grouped.items()):
            print(f"  expected {expected:22} predicted {predicted:22} {count}")

        print("\nSample failures")
        for failure in failures[:25]:
            print("-" * 80)
            print(f"ID: {failure['id']} ({failure['category']})")
            print(f"Claim: {failure['claim']}")
            print(f"Expected: {failure['expected']}")
            print(f"Predicted: {failure['predicted']}")
            print(f"Score: {round(failure.get('score', 0.0), 3)}")
            print(f"Confidence: {round(failure.get('confidence', 0.0), 3)}")
            if failure.get("reason"):
                print(f"Reason: {failure['reason']}")
            if failure.get("warning"):
                print(f"Warning: {failure['warning']}")
            if failure.get("unsupported_terms"):
                print(f"Unsupported terms: {', '.join(failure['unsupported_terms'])}")
            if failure.get("chunk_text"):
                print(f"Closest chunk: {failure['chunk_text']}")

    print("\n" + "=" * 80)
    print(f"Accuracy: {round(accuracy * 100, 2)}%")
    print(f"Pass threshold: {round(PASS_THRESHOLD * 100, 2)}%")
    print(f"Result: {'PASS' if accuracy >= PASS_THRESHOLD else 'FAIL'}")
    print("=" * 80)

    if accuracy < PASS_THRESHOLD:
        raise SystemExit(1)
    return accuracy


if __name__ == "__main__":
    run_benchmark()
