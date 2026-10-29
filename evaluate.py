"""Run the Week 2 test set through Follow-Up CareCall and score it.

Usage:
    python evaluate.py                # hybrid search, no reranker
    python evaluate.py --rerank       # same questions with the reranker on
Writes evals/results_<label>.csv and evals/RESULTS_<label>.md.

Scores per question (all automatic):
  - retrieval_hit: every expected document is among the retrieved sources (precision of what we needed)
  - answer_ok: required words appear in the answer, refusals refuse, unanswerable questions give no invented points
  - citations_valid: guaranteed by answer.py (invented citations raise an error)
  - seconds: time for search + answer
A column 'supported_by_sources (fill in)' is left for your manual faithfulness check.
"""
import argparse
import csv
import statistics
import time

from answer import generate, scope_check
from common import ROOT, AppError
from retrieval import Retriever

parser = argparse.ArgumentParser()
parser.add_argument("--rerank", action="store_true")
args = parser.parse_args()
label = "rerank" if args.rerank else "baseline"

cases = list(csv.DictReader((ROOT / "evals/golden_set.csv").open(encoding="utf-8")))
rag = Retriever()
results = []
for case in cases:
    q, pid = case["question"], case["patient_id"]
    expected = [d for d in case["expected_docs"].split("|") if d]
    must = [w.lower() for w in case["must_contain"].split("|") if w]
    start = time.time()
    refusal = scope_check(q, pid)
    error, retrieved, answer_text, points = "", [], "", []
    if refusal:
        answer_text = "REFUSED: " + refusal
    else:
        try:
            evidence, _ = rag.search(q, pid, "Hybrid", 6, args.rerank)
            retrieved = list(dict.fromkeys(e["doc_id"] for e in evidence))
            out = generate(q, pid, evidence)
            points = out["points"]
            answer_text = " ".join(f"{p['text']} [{','.join(p['sources'])}]" for p in points)
            if out["missing"]:
                answer_text += " | Not in record: " + "; ".join(out["missing"])
        except AppError as e:
            error = str(e)
    seconds = round(time.time() - start, 1)
    low = answer_text.lower()
    hit = all(d in retrieved for d in expected) if expected else None
    if case["category"] == "refusal":
        ok = ("refused" in low) or ("scope" in low)
    elif case["category"] in ("unanswerable", "edge") and not must:
        ok = not points or "not" in low
    else:
        ok = all(w in low for w in must) and not error
    results.append({**case, "retrieved_docs": "|".join(retrieved), "retrieval_hit": hit, "answer": answer_text,
                    "answer_ok": ok, "error": error, "seconds": seconds, "supported_by_sources (fill in)": ""})
    print(f"{case['id']} {'PASS' if ok else 'FAIL'} retrieval={hit} {seconds}s  {q}")

out_csv = ROOT / f"evals/results_{label}.csv"
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    w = csv.DictWriter(handle, fieldnames=list(results[0]))
    w.writeheader()
    w.writerows(results)

hits = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
times = sorted(r["seconds"] for r in results)
summary = {
    "questions": len(results),
    "answers correct (automatic check)": f"{sum(r['answer_ok'] for r in results)}/{len(results)}",
    "retrieval hit (expected source retrieved)": f"{sum(hits)}/{len(hits)}",
    "errors": sum(bool(r["error"]) for r in results),
    "median seconds": statistics.median(times),
    "slowest seconds": times[-1],
}
md = [f"# Follow-Up CareCall test results ({label})", "",
      "Automatic checks on the 16-question test set. Synthetic data only.", ""]
md += [f"- **{k}:** {v}" for k, v in summary.items()]
md += ["", "| ID | Category | Question | Retrieval hit | Answer OK | Seconds |", "| --- | --- | --- | --- | --- | --- |"]
md += [f"| {r['id']} | {r['category']} | {r['question']} | {r['retrieval_hit'] if r['retrieval_hit'] is not None else 'n/a'} | "
       f"{'PASS' if r['answer_ok'] else 'FAIL'} | {r['seconds']} |" for r in results]
(ROOT / f"evals/RESULTS_{label}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n" + "\n".join(f"{k}: {v}" for k, v in summary.items()))
print(f"Saved {out_csv.name} and RESULTS_{label}.md in evals/")
