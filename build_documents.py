"""Step 1: turn the 200-patient CSVs into readable documents with metadata.

Usage: python build_documents.py
Reads:  data/raw/synthea_csv_nov2021/csv/*.csv
Writes: data/processed/documents.jsonl  (one document per line)

Two kinds of document (the course pattern: content + structured metadata):
  1. patient_summary: one per patient. Allergies, active conditions, active medications,
     active care plans, and the most recent visits. Answers "what should I know before the call?"
  2. visit: one per encounter. What happened at that visit: diagnoses, medications,
     procedures, care plans, vaccines, labs and vitals.
Every document keeps the CSV file and row numbers it was built from, for citations.
"""
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/synthea_csv_nov2021/csv"
OUT = ROOT / "data/processed/documents.jsonl"
DATA_END = date(2021, 11, 19)  # last date in this Synthea sample; "active" means no stop date by then
RECENT_VISITS = 5


def read(name):
    with (RAW / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [(number, row) for number, row in enumerate(csv.DictReader(handle), 2)]


def day(ts):
    return (ts or "")[:10]


def clean_name(text):
    """Synthea adds digits to names (Miki234). Strip them for readability."""
    return "".join(ch for ch in (text or "") if not ch.isdigit()).strip()


def age_on(birth, when=DATA_END):
    b = date.fromisoformat(birth)
    return when.year - b.year - ((when.month, when.day) < (b.month, b.day))


patients = {r["Id"]: (n, r) for n, r in read("patients")}
panels = {r["PATIENT"]: dict(r, _row=n) for n, r in read("physician_panels")}
providers = {r["Id"]: r for _, r in read("providers")}
encounters = read("encounters")
enc_by_id = {r["Id"]: (n, r) for n, r in encounters}

TABLES = ["conditions", "medications", "procedures", "careplans", "immunizations", "observations", "allergies"]
by_encounter = defaultdict(lambda: defaultdict(list))
by_patient = defaultdict(lambda: defaultdict(list))
for table in TABLES:
    for n, r in read(table):
        by_patient[r["PATIENT"]][table].append((n, r))
        if r.get("ENCOUNTER"):
            by_encounter[r["ENCOUNTER"]][table].append((n, r))


def visit_header(r):
    prov = providers.get(r["PROVIDER"], {})
    reason = f" Reason: {r['REASONDESCRIPTION']}." if r.get("REASONDESCRIPTION") else ""
    return (f"Visit {r['Id']} on {day(r['START'])} ({r['ENCOUNTERCLASS']}): {r['DESCRIPTION']}.{reason}"
            f" Seen by {clean_name(prov.get('NAME', 'unknown provider'))}.")


def line_for(table, r):
    stop = f" to {day(r['STOP'])}" if r.get("STOP") else " (no stop date recorded)"
    if table == "conditions":
        return f"Diagnosis: {r['DESCRIPTION']}, from {day(r['START'])}{stop}."
    if table == "medications":
        why = f" for {r['REASONDESCRIPTION']}" if r.get("REASONDESCRIPTION") else ""
        return f"Medication: {r['DESCRIPTION']}{why}, from {day(r['START'])}{stop}."
    if table == "procedures":
        why = f" for {r['REASONDESCRIPTION']}" if r.get("REASONDESCRIPTION") else ""
        return f"Procedure: {r['DESCRIPTION']}{why} on {day(r['START'])}."
    if table == "careplans":
        why = f" for {r['REASONDESCRIPTION']}" if r.get("REASONDESCRIPTION") else ""
        return f"Care plan: {r['DESCRIPTION']}{why}, from {day(r['START'])}{stop}."
    if table == "immunizations":
        return f"Vaccine: {r['DESCRIPTION']} on {day(r['DATE'])}."
    if table == "observations":
        units = f" {r['UNITS']}" if r.get("UNITS") else ""
        return f"Lab or vital: {r['DESCRIPTION']} = {r['VALUE']}{units} on {day(r['DATE'])}."
    if table == "allergies":
        reaction = f"; reaction: {r['DESCRIPTION1']}" if r.get("DESCRIPTION1") else ""
        severity = f" ({r['SEVERITY1'].lower()})" if r.get("SEVERITY1") else ""
        return f"Allergy: {r['DESCRIPTION']} ({r.get('CATEGORY', '')}){reaction}{severity}, recorded {day(r['START'])}{stop}."
    raise ValueError(table)


SECTION = {"conditions": "Diagnoses", "medications": "Medications", "procedures": "Procedures",
           "careplans": "Care plans", "immunizations": "Vaccines", "observations": "Labs and vitals",
           "allergies": "Allergies"}

docs = []
for pid, (prow, p) in patients.items():
    panel = panels[pid]
    name = f"{clean_name(p['FIRST'])} {clean_name(p['LAST'])}"
    base = {"patient_id": pid, "patient_name": name, "physician": clean_name(panel["PHYSICIAN_NAME"])}

    # ---- visit documents
    visits = sorted((r for _, r in encounters if r["PATIENT"] == pid), key=lambda r: r["START"])
    for r in visits:
        n_enc = enc_by_id[r["Id"]][0]
        lines, sources = [visit_header(r)], {"encounters.csv": [n_enc]}
        for table in TABLES:
            rows = by_encounter[r["Id"]].get(table, [])
            if rows:
                lines.append(f"{SECTION[table]}:")
                lines += [line_for(table, x) for _, x in rows]
                sources[f"{table}.csv"] = [n for n, _ in rows]
        docs.append({"doc_id": f"{pid}-{r['Id']}", "text": "\n".join(lines),
                     "metadata": {**base, "doc_type": "visit", "encounter_id": r["Id"], "date": day(r["START"]),
                                  "encounter_class": r["ENCOUNTERCLASS"], "title": f"Visit {r['Id']} on {day(r['START'])}"},
                     "sources": sources})

    # ---- patient summary document
    mine = by_patient[pid]
    active = lambda table: [(n, x) for n, x in mine.get(table, []) if not x.get("STOP")]
    sources = {"patients.csv": [prow], "physician_panels.csv": [panel["_row"]]}
    lines = [f"Patient summary for {pid} ({name}). Sex: {p['GENDER']}. Born {p['BIRTHDATE']} "
             f"(age {age_on(p['BIRTHDATE'])} on {DATA_END}). Primary physician: {clean_name(panel['PHYSICIAN_NAME'])}."]
    allergies = mine.get("allergies", [])
    lines.append("Allergies:" if allergies else "Allergies: none recorded in the chart.")
    lines += [line_for("allergies", x) for _, x in allergies]
    for table, label in (("conditions", "Active diagnoses"), ("medications", "Active medications"),
                         ("careplans", "Active care plans")):
        rows = active(table)
        lines.append(f"{label} (no stop date):" if rows else f"{label}: none recorded.")
        lines += [line_for(table, x) for _, x in rows]
        if rows:
            sources[f"{table}.csv"] = [n for n, _ in rows]
    if allergies:
        sources["allergies.csv"] = [n for n, _ in allergies]
    recent = visits[-RECENT_VISITS:][::-1]
    lines.append(f"Most recent visits (newest first, last {len(recent)}):")
    lines += [visit_header(r) for r in recent]
    sources["encounters.csv"] = [enc_by_id[r["Id"]][0] for r in recent]
    lines.append("This chart has no appointment scheduling data, so future appointments are not known.")
    docs.append({"doc_id": f"{pid}-summary", "text": "\n".join(lines),
                 "metadata": {**base, "doc_type": "patient_summary", "encounter_id": "",
                              "date": day(recent[0]["START"]) if recent else "", "encounter_class": "",
                              "title": f"Patient summary for {pid}"},
                 "sources": sources})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as handle:
    for d in docs:
        handle.write(json.dumps(d, ensure_ascii=False) + "\n")
kinds = defaultdict(int)
for d in docs:
    kinds[d["metadata"]["doc_type"]] += 1
print(f"Wrote {len(docs):,} documents to {OUT.relative_to(ROOT)}: " + ", ".join(f"{k} {v:,}" for k, v in kinds.items()))
