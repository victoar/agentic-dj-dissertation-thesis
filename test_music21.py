from music21 import key as m21key

print("=" * 45)
print("TEST 3 — music21 Camelot Wheel")
print("=" * 45)

# Camelot Wheel — maps musical keys to compatibility groups.
# Adjacent numbers on the wheel are harmonically compatible.
# Same number, different letter (A/B) = relative major/minor.

CAMELOT = {
    # Major keys (B suffix)
    "C major":  "8B",  "G major":  "9B",  "D major":  "10B",
    "A major":  "11B", "E major":  "12B", "B major":  "1B",
    "F# major": "2B",  "Db major": "3B",  "Ab major": "4B",
    "Eb major": "5B",  "Bb major": "6B",  "F major":  "7B",
    # Minor keys (A suffix)
    "A minor":  "8A",  "E minor":  "9A",  "B minor":  "10A",
    "F# minor": "11A", "C# minor": "12A", "G# minor": "1A",
    "D# minor": "2A",  "A# minor": "3A",  "F minor":  "4A",
    "C minor":  "5A",  "G minor":  "6A",  "D minor":  "7A",
}

def camelot_compatible(key1: str, key2: str) -> bool:
    """Two keys are compatible if their Camelot positions differ by at most 1."""
    c1 = CAMELOT.get(key1)
    c2 = CAMELOT.get(key2)
    if not c1 or not c2:
        return False
    num1, letter1 = int(c1[:-1]), c1[-1]
    num2, letter2 = int(c2[:-1]), c2[-1]
    if letter1 == letter2:
        diff = abs(num1 - num2)
        return diff <= 1 or diff == 11   # wheel wraps: 12 -> 1
    else:
        return num1 == num2              # relative major/minor

# 1. Basic key parsing
k = m21key.Key("C")
print(f"\n[1] Key parsing  ✓  C major relative: {k.relative.tonicPitchNameWithCase}")

# 2. Compatibility checks
tests = [
    ("Ab major", "Eb major", True,  "adjacent on wheel"),
    ("Ab major", "F minor",  True,  "relative minor"),
    ("Ab major", "C# major", False, "incompatible — too far"),
    ("C major",  "G major",  True,  "adjacent on wheel"),
    ("C major",  "F# major", False, "tritone — incompatible"),
]

print(f"\n[2] Compatibility checks:")
all_pass = True
for k1, k2, expected, reason in tests:
    result  = camelot_compatible(k1, k2)
    status  = "✓" if result == expected else "✗"
    if result != expected:
        all_pass = False
    compatible = "compatible  " if result else "incompatible"
    print(f"      {status}  {k1:12} → {k2:12}  {compatible}  ({reason})")

# 3. Show Camelot positions for common keys
print(f"\n[3] Camelot positions:")
common = ["C major", "G major", "D major", "A minor", "E minor", "Ab major"]
for k in common:
    print(f"      {k:12}  →  {CAMELOT[k]}")

if all_pass:
    print("\n✓ Test 3 passed — music21 + Camelot Wheel working\n")
else:
    print("\n✗ Some checks failed — review logic above\n")