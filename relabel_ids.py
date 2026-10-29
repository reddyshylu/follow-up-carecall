"""Replace long Synthea IDs with short readable labels, in place.

Usage: python relabel_ids.py data/raw/synthea_csv_nov2021/csv

- Patients:   ptA, ptB, ... ptZ, ptAA, ... (ordered by physician panel, then name)
- Encounters: enaaa, enaab, enaac, ...  (ordered by patient, then visit date)
- Payers:     101, 102, 103, ...        (ordered by first visit date)
The original-to-new mapping is saved in id_map.csv so every label can be traced back.
Run once, right after make_200_patient_subset.py. Running it twice is refused.
"""
import csv
import string
import sys
from pathlib import Path

folder = Path(sys.argv[1])
TABLES = ["patients", "encounters", "conditions", "medications", "observations", "procedures",
          "careplans", "allergies", "immunizations", "physician_panels"]
if (folder / "id_map.csv").exists():
    sys.exit("id_map.csv already exists: IDs were already relabelled. Rebuild the subset first to run again.")


def read(name):
    with (folder / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, list(reader)


def write(name, header, rows):
    with (folder / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def letters(n, upper=True):
    """0 -> A, 25 -> Z, 26 -> AA (spreadsheet-style)."""
    abc = string.ascii_uppercase if upper else string.ascii_lowercase
    out = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        out = abc[r] + out
    return out


def triple(n):
    """0 -> aaa, 1 -> aab, ... (fixed three letters, 17,576 labels)."""
    abc = string.ascii_lowercase
    if n >= 26 ** 3:
        raise ValueError("More than 17,576 encounters; use four letters.")
    return abc[n // 676] + abc[(n // 26) % 26] + abc[n % 26]


data = {t: read(t) for t in TABLES}

# Patients, grouped by physician panel so each doctor's patients sit together
ph, prow = data["physician_panels"]
pat_h, pat_rows = data["patients"]
name_of = {r[pat_h.index("Id")]: (r[pat_h.index("LAST")], r[pat_h.index("FIRST")]) for r in pat_rows}
panel_order = sorted(prow, key=lambda r: (r[ph.index("PHYSICIAN_NAME")], name_of[r[ph.index("PATIENT")]]))
patient_map = {r[ph.index("PATIENT")]: "pt" + letters(i) for i, r in enumerate(panel_order)}

# Encounters, by patient label then start time
eh, erows = data["encounters"]
enc_sorted = sorted(erows, key=lambda r: (len(patient_map[r[eh.index("PATIENT")]]),
                                          patient_map[r[eh.index("PATIENT")]], r[eh.index("START")]))
encounter_map = {r[eh.index("Id")]: "en" + triple(i) for i, r in enumerate(enc_sorted)}

# Payers, by first appearance in time
first_seen = {}
for r in sorted(erows, key=lambda r: r[eh.index("START")]):
    first_seen.setdefault(r[eh.index("PAYER")], len(first_seen))
mh, mrows = data["medications"]
for r in sorted(mrows, key=lambda r: r[mh.index("START")]):
    first_seen.setdefault(r[mh.index("PAYER")], len(first_seen))
payer_map = {p: str(101 + i) for p, i in first_seen.items()}

COLUMN_MAPS = {"PATIENT": patient_map, "ENCOUNTER": encounter_map, "PAYER": payer_map}
missing = []
for t in TABLES:
    header, rows = data[t]
    targets = {i: COLUMN_MAPS[c] for i, c in enumerate(header) if c in COLUMN_MAPS}
    if t == "patients":
        targets[header.index("Id")] = patient_map
    if t == "encounters":
        targets[header.index("Id")] = encounter_map
    for r in rows:
        for i, mapping in targets.items():
            if r[i]:
                if r[i] not in mapping:
                    missing.append(f"{t}.{header[i]}: {r[i]}")
                else:
                    r[i] = mapping[r[i]]
    write(t, header, rows)

with (folder / "id_map.csv").open("w", encoding="utf-8", newline="") as handle:
    w = csv.writer(handle, lineterminator="\n")
    w.writerow(["TYPE", "NEW_ID", "ORIGINAL_SYNTHEA_ID"])
    for kind, mapping in (("patient", patient_map), ("encounter", encounter_map), ("payer", payer_map)):
        for old, new in sorted(mapping.items(), key=lambda kv: (len(kv[1]), kv[1])):
            w.writerow([kind, new, old])

print(f"Patients: {len(patient_map)} ({min(patient_map.values(), key=lambda x: (len(x), x))} to "
      f"{max(patient_map.values(), key=lambda x: (len(x), x))})")
print(f"Encounters: {len(encounter_map)} (enaaa to en{triple(len(encounter_map) - 1)})")
print(f"Payers: {len(payer_map)} (101 to {100 + len(payer_map)})")
if missing:
    sys.exit(f"{len(missing)} IDs had no mapping, first: {missing[:3]}")
print("Done. Mapping saved to id_map.csv")
