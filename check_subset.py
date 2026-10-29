"""Check the 200-patient subset: no deceased patients, no stray rows, every patient in one panel.

Usage: python check_subset.py data/raw/synthea_csv_nov2021/csv
Exits with an error if any check fails.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

folder = Path(sys.argv[1])
TABLES = ["encounters", "conditions", "medications", "observations", "procedures",
          "careplans", "allergies", "immunizations"]
DEATH_WORDS = ("death certification", "cause of death", "death certificate")


def rows(name):
    with (folder / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


fails = []
patients = rows("patients")
ids = {p["Id"] for p in patients}
dead = [p["Id"] for p in patients if p["DEATHDATE"].strip()]
results = [("Patients", len(patients), 200), ("Patients with a DEATHDATE", len(dead), 0)]

stray_total, death_total = 0, 0
for t in TABLES:
    data = rows(t)
    stray = [r for r in data if r["PATIENT"] not in ids]
    death = [r for r in data if any(w in (r.get("DESCRIPTION", "") or "").lower() for w in DEATH_WORDS)]
    stray_total += len(stray)
    death_total += len(death)
    if stray:
        fails.append(f"{t}.csv has {len(stray)} rows for patients not in patients.csv (first: {stray[0]['PATIENT']})")
    if death:
        fails.append(f"{t}.csv has {len(death)} death-related rows (first: {death[0]['DESCRIPTION']} for {death[0]['PATIENT']})")
results += [("Rows for patients not in patients.csv", stray_total, 0),
            ("Death-related records", death_total, 0)]

panels = rows("physician_panels")
per_patient = Counter(r["PATIENT"] for r in panels)
not_once = [p for p in ids if per_patient.get(p, 0) != 1]
extra = [p for p in per_patient if p not in ids]
results += [("Patients not in exactly one panel", len(not_once), 0),
            ("Panel rows for unknown patients", len(extra), 0)]

for label, got, want in results:
    ok = got == want
    if not ok and not any(label in f for f in fails):
        fails.append(f"{label}: got {got}, expected {want}")
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got}")

docs = Counter(r["PHYSICIAN_NAME"] for r in panels)
print(f"\nPhysicians: {len(docs)}; panel sizes: " + ", ".join(f"{d} {n}" for d, n in docs.most_common()))
if fails:
    print("\nFAILED:\n" + "\n".join(fails))
    sys.exit(1)
print("\nALL CHECKS PASSED")
