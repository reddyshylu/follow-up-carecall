# Follow-Up CareCall: Week 2 RAG Project

**One-liner:** Follow-Up CareCall helps nurses and care coordinators answer pre-call questions about one patient's chart, from 200 synthetic Synthea patient records, in a Streamlit app, with a target of 90% faithfulness (every answer point supported by its cited record) and a "not in the record" answer when the chart does not say, in under 10 seconds per answer.

All data is synthetic (Synthea). Never use real patient data with this app.

## Track and use case
- **Track:** code (LangChain + Python).
- **User:** a nurse or care coordinator preparing for a follow-up call. She picks her physician, picks a patient from that physician's panel, and asks questions.

## RAG framework
| Field | Decision |
| --- | --- |
| Use case | Pre-call questions about one patient (allergies, active medicines, last visit, care plans, labs, procedures), asked in a Streamlit web app |
| Corpus | Synthea November 2021 sample, cut to 200 living patients in 16 physician panels: 11,615 visits, 6,134 diagnoses, 6,617 medication records, 81,889 labs and vitals, 19,723 procedures, 885 care plans, 219 allergy records, 2,627 vaccines (English, CSV) |
| Ingestion and cleaning | `make_200_patient_subset.py` keeps living patients and assigns each to the physician seen most; `relabel_ids.py` replaces long IDs with readable ones (ptA, enaaa); `build_documents.py` turns CSV rows into readable text: one patient summary and one page per visit, each with metadata and the CSV rows it came from |
| Freshness | Static snapshot (data ends 2021-11-19). To refresh, re-run `build_documents.py` and `ingest.py --upload`; the upload replaces the namespace |
| Chunking and embedding | `RecursiveCharacterTextSplitter`, 1,000 characters with 150 overlap, giving 20,854 chunks; each chunk starts with a header naming the patient and visit. Embeddings: BAAI/bge-small-en-v1.5 (384 dimensions, free, runs locally); 1,000 characters stays well under its 512-token limit |
| Retrieve | Pinecone (namespace `carecall-200-v1`) with a metadata filter on the selected patient, plus BM25 keyword search over the same patient's chunks, fused with weighted Reciprocal Rank Fusion (0.5 / 0.5); top 6 passages plus the patient summary |
| Generate | Nebius Token Factory model through LangChain `ChatOpenAI`; every point must cite S1, S2...; invented citations are rejected in code |
| "I don't know" path | If the sources do not answer, the app says "not in the record"; another patient's ID is refused before any search; out-of-scope questions (for example, weather) are refused |

## How it works
1. `build_documents.py`: CSVs to 200 patient summaries + 11,615 visit documents with metadata (patient, physician, visit, date, type).
2. `ingest.py --upload`: chunk, embed with BGE, upload to Pinecone with metadata.
3. `retrieval.py`: hybrid search for ONE patient (Pinecone filter + BM25, RRF fusion).
4. `answer.py`: cited answer from Nebius; scope and citation checks.
5. `app.py`: Streamlit screen (physician, patient, question, cited answer, expandable sources).
6. `make_golden_set.py` and `evaluate.py`: 16-question test set and automatic scoring.

## Run it
1. Copy `.env.example` to `.env` and add `PINECONE_API_KEY`, `PINECONE_INDEX`, `NEBIUS_API_KEY`, `NEBIUS_MODEL`. Never commit `.env`.
2. `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`
3. `.venv\Scripts\python.exe build_documents.py`
4. `.venv\Scripts\python.exe ingest.py --upload` (about 40 minutes on a laptop CPU; `--resume` continues an interrupted upload)
5. `.venv\Scripts\python.exe -m streamlit run app.py --server.port 8503`

Or on Windows double-click `run_carecall.bat`, which does steps 3 to 5 and skips the upload once done.

## Optional test set
`evals/golden_set.csv` holds 16 sample questions (normal, edge, unanswerable, refusal). `python evaluate.py` scores them; this is optional for Week 2 and is the starting point for later evaluation.

## Known limits
- Care plans are labels only (for example, "Diabetes self management plan"); Synthea has no detailed instructions, so the app can name an active plan and its reason but not step-by-step instructions.
- No appointment scheduling data: the app cannot say when the next appointment is.
- "No allergies recorded" means none are documented, not that the patient has none.
- Retrieval is top-k, not a full chart review. The app reports documentation; it does not give medical advice.

## Data and tools
- Data: Synthea synthetic patient generator, November 2021 CSV sample (synthetic, no real patients).
- Libraries: LangChain (core, text splitters, OpenAI-compatible chat), sentence-transformers (BGE), Pinecone, rank-bm25, Streamlit.
- AI coding tools: Claude was used to design, write and debug the code.
