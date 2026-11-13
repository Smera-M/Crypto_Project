"""
Original Adaptive Encryption — attacker severity-prediction result.

The attacker observes:
1. Selected cipher identity
2. Encrypted record size

Reported attacker prediction accuracy: 93.33%
Majority-class baseline: 72.67%

This script reports the supplied experiment result. It does not independently
retrain or reproduce the attacker's classifier.
"""

ATTACKER_ACCURACY = 93.33
MAJORITY_BASELINE = 72.67

def main():
    print("ORIGINAL ADAPTIVE ENCRYPTION")
    print("-" * 40)

    print("Attacker can observe:")
    print("  1. Selected cipher identity")
    print("  2. Encrypted record size")

    print(f"\nAttacker severity-prediction accuracy: {ATTACKER_ACCURACY:.2f}%")
    print(f"Majority-class baseline: {MAJORITY_BASELINE:.2f}%")
    print(
        "Improvement over baseline: "
        f"{ATTACKER_ACCURACY - MAJORITY_BASELINE:.2f} percentage points"
    )

    print("\nInterpretation:")
    print(
        "The attacker predicts severity substantially better than the "
        "majority-class baseline, indicating that the observable cipher "
        "identity and encrypted record size reveal useful information."
    )

    print(
        "\nMetric note: 93.33% is classification accuracy, not a "
        "mutual-information leakage fraction."
    )


if __name__ == "__main__":
    main()
