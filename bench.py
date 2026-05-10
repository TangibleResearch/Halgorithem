from Halgorithem import Halgorithm

def run_benchmark():
    algo = Halgorithm(
        sentences_per_chunk=2,
        sentence_overlap=1
    )

    # -------------------------------------------------
    # TEST CLAIMS
    # -------------------------------------------------
    test_cases = [
        # SUPPORTED
        ("BASIC was developed in 1964", "SUPPORTED"),
        ("BASIC was used on time-sharing systems", "SUPPORTED"),
        ("BASIC was interpreted in early versions", "SUPPORTED"),

        # WEAK SUPPORT
        ("BASIC made programming easier for students", "WEAK_SUPPORT"),
        ("BASIC helped beginners learn programming", "WEAK_SUPPORT"),

        # HALLUCINATIONS
        ("BASIC was created by NASA", "HALLUCINATION"),
        ("BASIC was invented in Germany", "HALLUCINATION"),

        # CONTRADICTION
        ("BASIC was developed in 1972", "CONTRADICTION"),
    ]

    correct = 0
    total = len(test_cases)

    print("\nBenchmark Report")
    print("=" * 80)

    for claim, expected in test_cases:
        results = algo.compare_with_reasoning(
            truth_file_paths=[
                "sources/basic.txt",
                "sources/basic2.txt"
            ],
            ai_output=claim,
            threshold=0.30
        )

        result = results[0]
        predicted = result["status"]
        score = result["score"]

        if predicted == expected:
            correct += 1
        else:
            print("\n" + "-" * 80)
            print(f"Claim: {claim}")
            print(f"Expected: {expected}")
            print(f"Predicted: {predicted}")
            print(f"Score: {round(score, 3)}")

    accuracy = (correct / total) * 100

    print("\n" + "=" * 80)
    print(f"Accuracy: {round(accuracy, 2)}%")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
