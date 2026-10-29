r"""CareCall: follow-up call prep over 200 synthetic patients.
Run: .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8503
"""
import time

import streamlit as st

from answer import generate, scope_check
from common import APP_NAME, AppError
from retrieval import Retriever

st.set_page_config(page_title=APP_NAME, layout="wide")


@st.cache_resource
def engine():
    return Retriever()


try:
    rag = engine()
except AppError as error:
    st.error(str(error))
    st.stop()

# Patient directory from the chunk metadata
patients = {}
for pid, chunks in rag.by_patient.items():
    m = chunks[0]["metadata"]
    patients[pid] = {"name": m["patient_name"], "physician": m["physician"]}
physicians = sorted({p["physician"] for p in patients.values()})
label_sort = lambda pid: (len(pid), pid)

st.title(f"{APP_NAME}: follow-up call prep")
st.caption("Ask about one synthetic patient's record before a follow-up call. Every answer cites the record it came from. "
           "Synthetic Synthea data only; never use real patient data.")

with st.sidebar:
    st.header("Who are you calling?")
    physician = st.selectbox("Physician (your panel)", physicians)
    panel = sorted((pid for pid, p in patients.items() if p["physician"] == physician), key=label_sort)
    patient = st.selectbox("Patient", panel, format_func=lambda pid: f"{pid} · {patients[pid]['name']}")
    st.caption(f"{len(panel)} patients in this panel · {len(patients)} patients in total")
    st.divider()
    st.header("Search settings")
    strategy = st.selectbox("Search strategy", ["Hybrid", "Semantic", "Keyword"])
    k = st.slider("Extra passages (besides the patient summary)", 2, 10, 6)
    use_rerank = st.checkbox("Rerank with a cross-encoder", value=False,
                             help="Downloads a small model the first time (~90 MB).")
    use_llm = st.checkbox("Write an answer with Nebius", value=True, help="Off = show retrieved passages only.")
    show_cites = st.checkbox("Show citations and sources", value=False,
                             help="Off: just the answer. On: [S1] tags and the source records. "
                                  "Asking for a citation or source in your question also turns them on.")
    show_trace = st.checkbox("Show search details", value=False)

if st.session_state.get("patient") != patient:
    st.session_state.patient, st.session_state.history = patient, []

summary = next(c for c in rag.by_patient[patient] if c["metadata"]["doc_type"] == "patient_summary")
st.subheader(f"{patient} · {patients[patient]['name']}")
st.caption(summary["text"].split("\n")[1])  # the summary's first line: sex, birth date, physician

# Quick questions: a short button label -> the full question that is asked
examples = {"Allergies": "Does the patient have any allergies?",
            "Active medicines": "What active medications are recorded?",
            "Last visit": "When was the last visit and what was it for?",
            "Care plans": "What active care plans are there?"}
st.caption("Quick questions (click one, or type your own below):")
cols = st.columns(len(examples))
clicked = next((q for (label, q), col in zip(examples.items(), cols)
                if col.button(label, help=q, use_container_width=True)), None)
question = st.chat_input("Ask about this patient's record") or clicked

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        refusal = scope_check(question, patient)
        if refusal:
            st.warning(refusal)
            st.session_state.history.append({"q": question, "a": refusal})
            st.stop()
        try:
            start = time.time()
            evidence, trace = rag.search(question, patient, strategy, k, use_rerank)
            result = generate(question, patient, evidence) if use_llm else None
            elapsed = time.time() - start
        except AppError as error:
            st.error(str(error))
            st.stop()
        wants_cites = show_cites or any(w in question.lower() for w in ("citation", "cite", "source", "evidence"))
        lines = []
        if result:
            for p in result["points"]:
                tag = f" [{', '.join(p['sources'])}]" if wants_cites else ""
                lines.append(f"- {p['text']}{tag}")
            for m in result["missing"]:
                lines.append(f"- _Not in the record:_ {m}")
            if not lines:
                lines.append("- _Not found in this patient's record._")
        else:
            lines.append("_Answer generation is off; see the retrieved passages below._")
        text = "\n".join(lines)
        st.markdown(text)
        st.caption(f"{strategy} search{' + rerank' if use_rerank else ''} · {len(evidence)} passages · {elapsed:.1f} s")
        st.session_state.history.append({"q": question, "a": text})

        cited = {s for p in (result or {}).get("points", []) for s in p["sources"]}
        if wants_cites or not result:
            st.markdown("**Sources**")
        for e in (evidence if (wants_cites or not result) else []):
            mark = " (cited)" if e["label"] in cited else ""
            part = f" (part {e['chunk'] + 1})" if any(x["doc_id"] == e["doc_id"] and x is not e for x in evidence) else ""
            with st.expander(f"{e['label']}{mark} · {e['title']}{part} · {e['doc_type'].replace('_', ' ')}"):
                st.text(e["text"])
                rows = "; ".join(f"{f} rows {', '.join(map(str, r[:12]))}{' ...' if len(r) > 12 else ''}"
                                 for f, r in e["sources"].items() if r)
                st.caption(f"Built from: {rows}")
        if show_trace:
            st.json({name: ids for name, ids in trace.items()})
