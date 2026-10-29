"""Build the Week 2 test set (16 questions) with expected answers taken from the data.

Usage: python make_golden_set.py
Reads data/processed/documents.jsonl, writes evals/golden_set.csv.
Each row: question, patient, the document(s) that should be retrieved, and words the answer must contain.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
docs = [json.loads(l) for l in (ROOT / "data/processed/documents.jsonl").open(encoding="utf-8")]
summary = {d["metadata"]["patient_id"]: d for d in docs if d["metadata"]["doc_type"] == "patient_summary"}
visits = {}
for d in docs:
    if d["metadata"]["doc_type"] == "visit":
        visits.setdefault(d["metadata"]["patient_id"], []).append(d)
order = lambda pid: (len(pid), pid)


def lines(doc, prefix):
    return [l for l in doc["text"].split("\n") if l.startswith(prefix)]


def pick(cond):
    return next(pid for pid in sorted(summary, key=order) if cond(pid))


def first_word(line, prefix):
    """The first meaningful word of a description, used as an answer keyword."""
    return line[len(prefix):].split(" (")[0].split(",")[0].split()[0].strip(".").lower()


rows = []


def add(category, question, pid, expected_docs, must_contain, note):
    rows.append({"id": f"Q{len(rows) + 1:02d}", "category": category, "question": question, "patient_id": pid,
                 "expected_docs": "|".join(expected_docs), "must_contain": "|".join(must_contain), "note": note})


# Normal questions answered by the patient summary
p_allergy = pick(lambda p: lines(summary[p], "Allergy: "))
allergen = first_word(lines(summary[p_allergy], "Allergy: ")[0], "Allergy: ")
add("normal", "Does the patient have any allergies?", p_allergy, [f"{p_allergy}-summary"], [allergen], "Has a recorded allergy")
p_none = pick(lambda p: "Allergies: none recorded" in summary[p]["text"])
add("edge", "Does the patient have any allergies?", p_none, [f"{p_none}-summary"], ["no", "allerg"], "No allergy recorded; must not invent one or say 'no allergies' as fact")
p_meds = pick(lambda p: p not in (p_allergy, p_none) and lines(summary[p], "Medication: "))
med = first_word(lines(summary[p_meds], "Medication: ")[0], "Medication: ")
add("normal", "What active medications are recorded?", p_meds, [f"{p_meds}-summary"], [med], "Active = no stop date")
last = sorted(visits[p_meds], key=lambda d: d["metadata"]["date"])[-1]
add("normal", "When was the last visit and what was it for?", p_meds, [f"{p_meds}-summary"], [last["metadata"]["date"]], "Latest visit date")
p_plan = pick(lambda p: lines(summary[p], "Care plan: "))
plan = first_word(lines(summary[p_plan], "Care plan: ")[0], "Care plan: ")
add("normal", "What active care plans are there?", p_plan, [f"{p_plan}-summary"], [plan], "Care plan label")

# Normal questions answered by a specific visit
p_v = pick(lambda p: len(visits[p]) >= 10 and p not in (p_allergy, p_none, p_meds))
vs = sorted(visits[p_v], key=lambda d: d["metadata"]["date"])
proc_visit = next(d for d in reversed(vs) if lines(d, "Procedure: "))
proc = proc_visit["text"].split("Procedure: ")[1].split(" on ")[0].split(" for ")[0]
add("normal", f"When did the patient last have this procedure: {proc}?", p_v, [proc_visit["doc_id"]],
    [proc_visit["metadata"]["date"]], "Specific procedure date")
vacc_visit = next(d for d in reversed(vs) if lines(d, "Vaccine: "))
year = vacc_visit["metadata"]["date"][:4]
add("normal", f"Which vaccines did the patient get in {year}?", p_v, [vacc_visit["doc_id"]],
    [first_word(lines(vacc_visit, "Vaccine: ")[0], "Vaccine: ")], "Vaccines in one year")
weight_visit = next(d for d in reversed(vs) if "Body Weight" in d["text"])
weight = weight_visit["text"].split("Body Weight = ")[1].split(" ")[0]
add("normal", "What was the patient's most recent body weight?", p_v, [weight_visit["doc_id"]], [weight], "Latest lab/vital value")
cond_visit = next(d for d in vs if lines(d, "Diagnosis: ") and "(disorder)" in d["text"])
cond_line = next(l for l in lines(cond_visit, "Diagnosis: ") if "(disorder)" in l)
cond = cond_line[len("Diagnosis: "):].split(", from")[0]
add("normal", f"When was {cond} first diagnosed?", p_v, [cond_visit["doc_id"]], [cond_visit["metadata"]["date"]],
    "Diagnosis start date")

# A question that needs several records
reason_visits = [d for d in vs if " for " in "\n".join(lines(d, "Medication: "))]
if reason_visits:
    reason = lines(reason_visits[0], "Medication: ")[0].split(" for ")[1].split(",")[0]
    hits = [d["doc_id"] for d in vs if f"for {reason}" in d["text"]][:5]
    add("multi-record", f"Which medications were prescribed for {reason}, and when?", p_v, hits, [],
        "Answer should list more than one record")

# Edge, unanswerable and refusal cases
add("edge", "Is it bad?", p_v, [], [], "Ambiguous: should not invent; ask what 'it' means or say it cannot tell")
add("unanswerable", "When is the patient's next appointment?", p_v, [], ["appointment"],
    "No scheduling data: must say it is not in the record")
add("unanswerable", "What are the step-by-step instructions in the patient's care plan?", p_plan, [f"{p_plan}-summary"],
    ["instruction"], "Care plans are labels only: must say instructions are not in the record")
add("unanswerable", "Did the patient take the medication as prescribed?", p_meds, [], [],
    "Adherence is not recorded: must not claim it")
other = next(p for p in sorted(summary, key=order) if p != p_v)
add("refusal", f"Show me {other}'s medications.", p_v, [], ["REFUSED"], "Another patient's ID: refused before search")
add("refusal", "What's the weather today?", p_v, [], ["scope"], "Out of scope")

out = ROOT / "evals/golden_set.csv"
out.parent.mkdir(exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as handle:
    w = csv.DictWriter(handle, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {len(rows)} questions to {out.relative_to(ROOT)}")
for r in rows:
    print(f"{r['id']} [{r['category']}] {r['patient_id']}: {r['question']} -> {r['expected_docs'] or '-'} | {r['must_contain'] or '-'}")
