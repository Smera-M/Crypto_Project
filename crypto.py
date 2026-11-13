"""
Runtime encryption with a severity-driven cipher policy.

The cipher policy lives in policy.py, so this runtime code and the leakage
analysis in leakage.py always use the same definition.

KEY HANDLING IS DEMONSTRATIVE. Keys are held in InMemoryKeyStore, separate from
the stored record, which carries only a random key_id. A real deployment needs a
key-management service or HSM, access control on keys, rotation and key
destruction. Each message gets a fresh random key and a library-generated
random nonce, so a (key, nonce) pair is never reused.

AUTHENTICATION. AES-GCM and ChaCha20-Poly1305 are authenticated: decryption
raises instead of returning data if the ciphertext, tag, nonce or key is wrong.
decrypt_message also rejects a record whose cipher_id disagrees with its
algorithm or with the length of the key it points to.

INVALID SEVERITY. A missing or unknown severity falls back to
policy.FALLBACK_TIER (the strongest tier), is flagged in the record and raises
a warning. Pass strict=True to raise instead.
"""

import time
import warnings

from Crypto.Cipher import AES, ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

from policy import (
    CIPHER_SPECS,
    DEFAULT_POLICY,
    choose_cipher,
    resolve_severity,
)


class InMemoryKeyStore:
    """DEMONSTRATIVE key store. Replace with a KMS or HSM in any real use."""

    def __init__(self):
        self._keys = {}

    def put(self, key):
        while True:
            key_id = get_random_bytes(8).hex()
            if key_id not in self._keys:
                break
        self._keys[key_id] = bytes(key)
        return key_id

    def get(self, key_id):
        return self._keys[key_id]       # KeyError for an unknown id

    def __len__(self):
        return len(self._keys)


def _new_cipher(algorithm, key, nonce=None):
    if algorithm == "AES-GCM":
        if nonce is None:
            return AES.new(key, AES.MODE_GCM)
        return AES.new(key, AES.MODE_GCM, nonce=nonce)
    if algorithm == "ChaCha20-Poly1305":
        if nonce is None:
            return ChaCha20_Poly1305.new(key=key)
        return ChaCha20_Poly1305.new(key=key, nonce=nonce)
    raise ValueError("Unsupported cipher algorithm")


def encrypt_message(message, severity, key_store, policy=DEFAULT_POLICY,
                    rng=None, strict=False, cipher_id=None):
    """Encrypt `message` under the cipher the policy picks for `severity`.

    cipher_id forces a specific cipher (for tests and benchmarks only).
    The returned record contains no key material.
    """
    tier, used_fallback = resolve_severity(severity)
    if used_fallback:
        if strict:
            raise ValueError(f"Unknown severity: {severity!r}")
        warnings.warn(
            f"Unknown severity {severity!r}: using fallback tier '{tier}'",
            RuntimeWarning, stacklevel=2)

    if cipher_id is None:
        cipher_id = choose_cipher(tier, policy, rng)
    elif cipher_id not in CIPHER_SPECS:
        raise ValueError(f"Unknown cipher: {cipher_id}")

    algorithm, key_size = CIPHER_SPECS[cipher_id]
    key = get_random_bytes(key_size)
    cipher = _new_cipher(algorithm, key)

    ciphertext, tag = cipher.encrypt_and_digest(message.encode("utf-8"))

    return {
        "severity": tier,                 # internal; not observer-visible
        "fallback_used": used_fallback,   # internal
        "policy": policy,                 # internal
        "cipher_id": cipher_id,           # observer-visible
        "algorithm": algorithm,
        "key_id": key_store.put(key),
        "nonce": cipher.nonce,
        "ciphertext": ciphertext,
        "tag": tag,
    }


def decrypt_message(record, key_store):
    algorithm = record["algorithm"]
    spec = CIPHER_SPECS.get(record["cipher_id"])
    if spec is None or spec[0] != algorithm:
        raise ValueError("cipher_id does not match the algorithm")

    key = key_store.get(record["key_id"])
    if len(key) != spec[1]:
        raise ValueError("key length does not match cipher_id")

    cipher = _new_cipher(algorithm, key, nonce=record["nonce"])
    plaintext = cipher.decrypt_and_verify(record["ciphertext"], record["tag"])
    return plaintext.decode("utf-8")


def observer_view(record):
    """Fields an observer with read access to the stored log can see.

    Matches the leakage model: the cipher identifier is visible; the key, the
    internal severity label and the policy are not.
    """
    return {k: record[k]
            for k in ("cipher_id", "key_id", "nonce", "ciphertext", "tag")}


def main():
    store = InMemoryKeyStore()
    message = (
        "Dynamic Encryption Project: "
        "confidential sample message."
    )

    print(f"Dynamic Encryption Demonstration (policy: {DEFAULT_POLICY})\n")

    for severity in ["Low", "Medium", "High", "Critical"]:
        start = time.perf_counter()
        record = encrypt_message(message, severity, store)
        elapsed_ms = (time.perf_counter() - start) * 1000

        recovered = decrypt_message(record, store)

        print(f"Severity: {severity}")
        print(f"Cipher identifier: {record['cipher_id']}")
        print(f"Algorithm: {record['algorithm']}")
        print(f"Key size: {len(store.get(record['key_id'])) * 8} bits")
        print(f"Ciphertext bytes: {len(record['ciphertext'])}")
        print(f"Encryption time: {elapsed_ms:.4f} ms")
        print(f"Decryption successful: {recovered == message}")
        print("-" * 45)

    print("\nNote: keys are in an in-memory demo key store, not a real "
          "key-management service.")


if __name__ == "__main__":
    main()
