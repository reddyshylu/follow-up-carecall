"""Make a 200-patient, physician-panel subset of the Synthea Nov 2021 sample.

Usage (from this project folder):
    python make_200_patient_subset.py <source_csv_folder> data/raw/synthea_csv_nov2021/csv

Rules:
- Only living patients (no DEATHDATE): nobody makes a follow-up call to a deceased patient.
- Primary physician = the provider the patient saw most often for wellness, ambulatory
  or outpatient visits.
- Physicians are taken largest panel first until 200 patients are reached; the last
  panel is trimmed so the total is exactly 200.
- Every row for the chosen patients is kept unchanged (values copied as text, so the
  app's source checks still match the CSV rows).
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

TARGET = 200
TABLES = ["encounters", "conditions", "medications", "observations", "procedures",
          "careplans", "allergies", "immunizations"]

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)


def read(name):
    with (src / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, list(reader)


p_head, p_rows = read("patients")
alive = {r[p_head.index("Id")] for r in p_rows if not r[p_head.index("DEATHDATE")].strip()}

e_head, e_rows = read("encounters")
ip, iv, ic = e_head.index("PATIENT"), e_head.index("PROVIDER"), e_head.index("ENCOUNTERCLASS")
visits = defaultdict(Counter)
for r in e_rows:
    if r[ip] in alive and r[ic] in ("wellness", "ambulatory", "outpatient"):
        visits[r[ip]][r[iv]] += 1
primary = {pid: c.most_common(1)[0] for pid, c in visits.items()}  # pid -> (provider, n)

panels = defaultdict(list)
for pid, (prov, n) in primary.items():
    panels[prov].append((n, pid))
order = sorted(panels, key=lambda prov: (-len(panels[prov]), prov))

# Take whole panels until the target is reached, then trim the largest panels one
# patient at a time (fewest primary visits first), so no physician is left with a stub panel.
picked = {}
for prov in order:
    if sum(len(m) for m in picked.values()) >= TARGET:
        break
    picked[prov] = sorted(panels[prov], key=lambda x: (-x[0], x[1]))
while sum(len(m) for m in picked.values()) > TARGET:
    biggest = max(picked, key=lambda prov: (len(picked[prov]), prov))
    picked[biggest].pop()

panel_rows = [(pid, prov, n, sum(visits[pid].values()))
              for prov, members in picked.items() for n, pid in members]
keep = {row[0] for row in panel_rows}

pv_head, pv_rows = read("providers")
prov_info = {r[pv_head.index("Id")]: r for r in pv_rows}


def write(name, header, rows):
    with (dst / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


counts = {"patients": write("patients", p_head, [r for r in p_rows if r[p_head.index("Id")] in keep])}
used_prov, used_org = set(), set()
for t in TABLES:
    head, rows = (e_head, e_rows) if t == "encounters" else read(t)
    sub = [r for r in rows if r[head.index("PATIENT")] in keep]
    if t == "encounters":
        used_prov |= {r[iv] for r in sub}
        used_org |= {r[head.index("ORGANIZATION")] for r in sub}
    counts[t] = write(t, head, sub)
counts["providers"] = write("providers", pv_head, [r for r in pv_rows if r[pv_head.index("Id")] in used_prov])
o_head, o_rows = read("organizations")
counts["organizations"] = write("organizations", o_head, [r for r in o_rows if r[o_head.index("Id")] in used_org])

name_i, spec_i = pv_head.index("NAME"), pv_head.index("SPECIALITY")
with (dst / "physician_panels.csv").open("w", encoding="utf-8", newline="") as handle:
    w = csv.writer(handle, lineterminator="\n")
    w.writerow(["PATIENT", "PROVIDER", "PHYSICIAN_NAME", "SPECIALITY", "PRIMARY_VISITS", "ALL_CLINIC_VISITS"])
    for pid, prov, n, total in panel_rows:
        info = prov_info[prov]
        w.writerow([pid, prov, info[name_i], info[spec_i], n, total])

per_doc = Counter(prov_info[p][name_i] for _, p, _, _ in panel_rows)
print(f"Patients: {len(keep)} (living only) across {len(per_doc)} physicians")
for doc, n in per_doc.most_common():
    print(f"  {doc}: {n}")
for k, v in counts.items():
    print(f"{k}.csv: {v:,} rows")
