"""
Single source of truth for severity tiers, ciphers and cipher-selection policies.

Imported by crypto.py (runtime encryption) and leakage.py (analysis) so the two
cannot silently drift apart. Standard library only.

A policy maps each severity tier to a probability distribution over cipher
identifiers. Randomisation may only go UPWARD: no tier may ever receive a
cipher weaker than MIN_STRENGTH[tier]. This is checked when the module loads.

To change which policy the runtime system deploys, change DEFAULT_POLICY, then
re-run leakage.py so the analysis and the runtime describe the same system.
"""

import random

SEVERITIES = ["Low", "Medium", "High", "Critical"]

AES128 = "AES-128-GCM"
AES192 = "AES-192-GCM"
AES256 = "AES-256-GCM"
CHACHA = "ChaCha20-Poly1305"
CIPHERS = [AES128, AES192, AES256, CHACHA]

# Runtime parameters for each cipher identifier: (algorithm, key size in bytes).
CIPHER_SPECS = {
    AES128: ("AES-GCM", 16),
    AES192: ("AES-GCM", 24),
    AES256: ("AES-GCM", 32),
    CHACHA: ("ChaCha20-Poly1305", 32),
}

# Modelling assumption: ChaCha20 is treated as the same strength class as a
# 256-bit AES key. This is not a claim that the algorithms are identical in
# every security property or implementation context.
STRENGTH = {AES128: 1, AES192: 2, AES256: 3, CHACHA: 3}

# Weakest cipher strength each tier may receive. This is our operationalisation
# of the base paper's premise that more severe threats get stronger encryption;
# the paper gives no numeric minimum.
MIN_STRENGTH = {"Low": 1, "Medium": 2, "High": 3, "Critical": 3}


def _policy(*rows):
    return dict(zip(SEVERITIES, rows))


UNIFORM_256 = _policy(*[{AES256: 1.0} for _ in SEVERITIES])

DETERMINISTIC = _policy(
    {AES128: 1.0}, {AES192: 1.0}, {AES256: 1.0}, {CHACHA: 1.0})

MERGE_HIGH_CRITICAL = _policy(
    {AES128: 1.0}, {AES192: 1.0}, {AES256: 1.0}, {AES256: 1.0})

# Weights as stated in Lor et al. (2026), Section 3.4 (checked against the
# paper text). The paper says only "AES" and "ChaCha20"; naming the modes
# GCM / Poly1305 is our own modelling assumption.
BASE_PAPER = _policy(
    {AES128: 0.85, AES192: 0.15},
    {AES192: 0.80, AES256: 0.20},
    {CHACHA: 0.70, AES256: 0.30},
    {AES256: 0.75, CHACHA: 0.25},
)


def mix_policies(a, b, lam):
    """Randomise between two policies: (1 - lam) * a + lam * b."""
    out = {}
    for severity in SEVERITIES:
        row = {}
        for cipher in CIPHERS:
            p = ((1.0 - lam) * a[severity].get(cipher, 0.0)
                 + lam * b[severity].get(cipher, 0.0))
            if p > 0:
                row[cipher] = p
        out[severity] = row
    return out


POLICIES = {
    "Base paper (weighted random)": BASE_PAPER,
    "Deterministic (4 ciphers)": DETERMINISTIC,
    "Merge High+Critical": MERGE_HIGH_CRITICAL,
    "Merge + noise (lam=0.25)": mix_policies(MERGE_HIGH_CRITICAL, UNIFORM_256, 0.25),
    "Merge + noise (lam=0.50)": mix_policies(MERGE_HIGH_CRITICAL, UNIFORM_256, 0.50),
    "Uniform AES-256 (endpoint)": UNIFORM_256,
}

# The policy crypto.py deploys. The base-paper policy is the experimental
# baseline; switch this once the proposed defence has been chosen.
DEFAULT_POLICY = "Base paper (weighted random)"


def validate_policy(name, policy):
    """Raise ValueError unless the policy is complete, normalised and upward-only."""
    for severity in SEVERITIES:
        if severity not in policy:
            raise ValueError(f"{name}: missing tier '{severity}'")
        row = policy[severity]
        if abs(sum(row.values()) - 1.0) > 1e-9:
            raise ValueError(f"{name}/{severity}: probabilities must sum to 1")
        for cipher, prob in row.items():
            if cipher not in CIPHERS:
                raise ValueError(f"{name}/{severity}: unknown cipher '{cipher}'")
            if prob < 0:
                raise ValueError(f"{name}/{severity}: negative probability")
            if prob > 0 and STRENGTH[cipher] < MIN_STRENGTH[severity]:
                raise ValueError(
                    f"{name}/{severity}: {cipher} is below the tier minimum")


def _validate_registry():
    for name, policy in POLICIES.items():
        validate_policy(name, policy)
    if DEFAULT_POLICY not in POLICIES:
        raise ValueError(f"DEFAULT_POLICY '{DEFAULT_POLICY}' is not registered")


_validate_registry()

# Used when the severity is missing, invalid or the classifier fails. It is the
# strongest tier, so a failure can never silently select weaker encryption.
FALLBACK_TIER = "Critical"


def resolve_severity(severity):
    """Return (tier, used_fallback). Anything that is not a known tier falls back."""
    if isinstance(severity, str) and severity in SEVERITIES:
        return severity, False
    return FALLBACK_TIER, True

_SYSTEM_RNG = random.SystemRandom()


def choose_cipher(severity, policy=DEFAULT_POLICY, rng=None):
    """Draw a cipher identifier for a severity tier under a named policy.

    rng: optional random.Random instance (use a seeded one for reproducible
    experiments). Defaults to the operating system's random source.
    """
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}")
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    row = POLICIES[policy][severity]
    ciphers = list(row)
    weights = [row[c] for c in ciphers]
    return (rng or _SYSTEM_RNG).choices(ciphers, weights=weights, k=1)[0]


def describe_policies():
    """Print every registered policy; paste this into the report."""
    print(f"Deployed default policy: {DEFAULT_POLICY}")
    print(f"Fallback tier (invalid severity or classifier failure): {FALLBACK_TIER}\n")
    for name, policy in POLICIES.items():
        mark = "  <-- deployed" if name == DEFAULT_POLICY else ""
        print(f"{name}{mark}")
        for severity in SEVERITIES:
            row = ", ".join(f"{c} {p:.0%}" for c, p in policy[severity].items())
            print(f"    {severity:9s} -> {row}")
        print()


if __name__ == "__main__":
    describe_policies()
