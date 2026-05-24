from collections import Counter, defaultdict, deque

from Halgorithem import Halgorithm


SOURCE_FILES = [
    "sources/basic.txt",
    "sources/basic2.txt",
]


def _numbered(prefix, items, expected):
    return [
        {
            "id": f"{prefix}-{i:03d}",
            "claim": claim,
            "expected": expected,
        }
        for i, claim in enumerate(items, 1)
    ]


def build_test_cases():
    supported_subjects = [
        "BASIC",
        "The BASIC programming language",
        "The language BASIC",
        "The BASIC language",
        "The programming language BASIC",
    ]
    supported = []
    for subject in supported_subjects:
        supported.extend([
            f"{subject} is a programming language.",
            f"{subject} was created in 1964.",
            f"{subject} was created by John G. Kemeny.",
            f"{subject} was created by Thomas E. Kurtz.",
            f"{subject} was created at Dartmouth College.",
            f"{subject} was developed at Dartmouth College.",
            f"{subject} was designed for students.",
            f"{subject} allowed students to write programs easily.",
            f"{subject} was used on time-sharing systems.",
            f"{subject} was used in early computing environments.",
            f"{subject} was interpreted in many early implementations.",
            f"{subject} used interpreters in early systems.",
            f"{subject} allowed multiple users to interact with a computer simultaneously.",
            f"{subject} became popular because of time-sharing.",
            f"{subject} used interpreters rather than compilers.",
            f"{subject} was designed without requiring deep technical knowledge.",
            f"{subject} let students write programs without deep technical knowledge.",
            f"{subject} was widely used on time-sharing systems.",
            f"{subject} existed in early computing environments.",
            f"{subject} had early interpreted implementations.",
        ])

    weak_templates = [
        "BASIC helped beginners learn programming.",
        "BASIC helped students learn programming.",
        "BASIC made programming easier for beginners.",
        "BASIC made programming easy for students.",
        "BASIC was important for beginner programmers.",
        "BASIC was significant for programming education.",
        "BASIC was influential in early programming education.",
        "BASIC helped make computing easier for students.",
        "BASIC made time-sharing more useful for students.",
        "BASIC helped popularize interactive computing.",
    ]
    weak = []
    for i in range(10):
        weak.extend(weak_templates)

    hallucinated_creators = [
        "NASA",
        "MIT",
        "IBM",
        "Microsoft",
        "Apple",
        "Google",
        "Harvard",
        "Stanford",
        "Bell Labs",
        "Alan Turing",
        "Grace Hopper",
        "Ada Lovelace",
        "Edsger Dijkstra",
        "Dennis Ritchie",
        "Ken Thompson",
        "Tim Berners-Lee",
        "Guido van Rossum",
        "Bjarne Stroustrup",
        "Linus Torvalds",
        "Yale University",
    ]
    hallucinated_places = [
        "Germany",
        "France",
        "Japan",
        "Canada",
        "California",
        "Cambridge",
        "Princeton",
        "Oxford",
        "London",
        "Silicon Valley",
    ]
    hallucinations = []
    for creator in hallucinated_creators:
        hallucinations.extend([
            f"BASIC was created by {creator}.",
            f"BASIC was invented by {creator}.",
            f"The BASIC programming language came from {creator}.",
        ])
    for place in hallucinated_places:
        hallucinations.extend([
            f"BASIC was invented in {place}.",
            f"BASIC was developed in {place}.",
            f"The BASIC programming language originated in {place}.",
            f"Early BASIC systems were built in {place}.",
        ])

    contradiction_years = [
        1959, 1960, 1961, 1962, 1963,
        1965, 1966, 1967, 1968, 1969,
        1970, 1971, 1972, 1973, 1974,
        1975, 1976, 1977, 1978, 1979,
    ]
    contradictions = []
    for year in contradiction_years:
        contradictions.extend([
            f"BASIC was created in {year}.",
            f"BASIC was developed in {year}.",
            f"The BASIC programming language was created in {year}.",
            f"The BASIC programming language was developed in {year}.",
            f"BASIC was created at Dartmouth College in {year}.",
        ])

    denials = []
    for creator in hallucinated_creators:
        denials.extend([
            f"BASIC was not created by {creator}.",
            f"BASIC was never invented by {creator}.",
            f"BASIC was not developed by {creator}.",
        ])
    for place in hallucinated_places:
        denials.extend([
            f"BASIC was not invented in {place}.",
            f"BASIC was never developed in {place}.",
            f"BASIC did not originate in {place}.",
            f"BASIC was not created in {place}.",
        ])

    return (
        _numbered("supported", supported, "SUPPORTED")
        + _numbered("weak", weak, "WEAK_SUPPORT")
        + _numbered("hallucination", hallucinations, "HALLUCINATION")
        + _numbered("contradiction", contradictions, "CONTRADICTION")
        + _numbered("denial", denials, "UNVERIFIABLE_DENIAL")
    )


def run_benchmark():
    algo = Halgorithm(
        sentences_per_chunk=2,
        sentence_overlap=1,
    )
    source_docs = algo.load_files(SOURCE_FILES)
    test_cases = build_test_cases()
    expected_by_claim = defaultdict(deque)
    for case in test_cases:
        expected_by_claim[algo.clean_text(case["claim"])].append(case)

    ai_output = "\n".join(case["claim"] for case in test_cases)
    results = algo.compare_to_docs(
        truth_docs=source_docs,
        ai_output=ai_output,
        threshold=0.30,
    )

    correct = 0
    failures = []
    predicted_counts = Counter()
    expected_counts = Counter(case["expected"] for case in test_cases)
    category_totals = Counter(case["id"].split("-", 1)[0] for case in test_cases)
    category_correct = Counter()

    print("\nBenchmark Report")
    print("=" * 80)
    print(f"Test cases: {len(test_cases)}")
    print(f"Verifier results: {len(results)}")

    if len(results) != len(test_cases):
        print("\nWARNING: result count differs from test-case count.")

    for result in results:
        claim = result["claim"]
        matching_cases = expected_by_claim.get(claim)
        if not matching_cases:
            failures.append({
                "id": "unknown",
                "claim": claim,
                "expected": "NO_MATCHING_TEST_CASE",
                "predicted": result["status"],
                "score": result.get("score", 0.0),
                "reason": result.get("reason", ""),
            })
            continue
        case = matching_cases.popleft()

        expected = case["expected"]
        predicted = result["status"]
        predicted_counts[predicted] += 1

        category = case["id"].split("-", 1)[0]
        if predicted == expected:
            correct += 1
            category_correct[category] += 1
        else:
            failures.append({
                "id": case["id"],
                "claim": claim,
                "expected": expected,
                "predicted": predicted,
                "score": result.get("score", 0.0),
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason", ""),
                "unsupported_terms": result.get("unsupported_terms", []),
                "chunk_text": result.get("chunk_text", ""),
            })

    for remaining_cases in expected_by_claim.values():
        for case in remaining_cases:
            failures.append({
                "id": case["id"],
                "claim": algo.clean_text(case["claim"]),
                "expected": case["expected"],
                "predicted": "NO_RESULT",
                "score": 0.0,
                "confidence": 0.0,
                "reason": "Claim was not returned by the verifier.",
            })

    missed = len(test_cases) - len(results)
    total_correct = correct - max(missed, 0)
    accuracy = (total_correct / len(test_cases)) * 100

    print("\nExpected labels")
    for status, count in sorted(expected_counts.items()):
        print(f"  {status:22} {count}")

    print("\nPredicted labels")
    for status, count in sorted(predicted_counts.items()):
        print(f"  {status:22} {count}")

    print("\nAccuracy by category")
    for category, total in sorted(category_totals.items()):
        category_accuracy = (category_correct[category] / total) * 100
        print(f"  {category:15} {category_correct[category]:3d}/{total:<3d} {category_accuracy:6.2f}%")

    if failures:
        grouped = defaultdict(int)
        for failure in failures:
            grouped[(failure["expected"], failure["predicted"])] += 1

        print("\nFailure summary")
        for (expected, predicted), count in sorted(grouped.items()):
            print(f"  expected {expected:22} predicted {predicted:22} {count}")

        print("\nSample failures")
        for failure in failures[:25]:
            print("-" * 80)
            print(f"ID: {failure['id']}")
            print(f"Claim: {failure['claim']}")
            print(f"Expected: {failure['expected']}")
            print(f"Predicted: {failure['predicted']}")
            print(f"Score: {round(failure.get('score', 0.0), 3)}")
            print(f"Confidence: {round(failure.get('confidence', 0.0), 3)}")
            if failure.get("reason"):
                print(f"Reason: {failure['reason']}")
            if failure.get("unsupported_terms"):
                print(f"Unsupported terms: {', '.join(failure['unsupported_terms'])}")
            if failure.get("chunk_text"):
                print(f"Closest chunk: {failure['chunk_text']}")

    print("\n" + "=" * 80)
    print(f"Accuracy: {round(accuracy, 2)}%")
    print("=" * 80)

    return accuracy


if __name__ == "__main__":
    run_benchmark()
