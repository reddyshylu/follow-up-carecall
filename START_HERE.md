# Follow-Up CareCall (Week 2 RAG)

Follow-Up CareCall helps a nurse or care coordinator prepare for a follow-up call. She picks her physician,
picks a patient from that physician's panel, and asks questions. Every answer cites the record it came from.
Data: 200 living synthetic Synthea patients in 16 physician panels. Never use real patient data.

## How it works (course pattern)
1. `build_documents.py`: CSVs -> 200 patient summaries + 11,615 visit documents, each with metadata
   (patient, physician, visit, date, type) and the CSV rows it came from.
2. `ingest.py`: split into 20,854 chunks (1,000 characters, 150 overlap) -> BGE-small embeddings ->
   Pinecone namespace `carecall-200-v1` with metadata. The old project's namespace is not touched.
3. `retrieval.py`: hybrid search for ONE patient: Pinecone with a patient metadata filter + BM25 keyword
   search, fused with weighted Reciprocal Rank Fusion. The patient summary is always included.
4. `answer.py`: Nebius writes short points; every point must cite S1, S2...; invented citations are rejected;
   other patients' IDs are refused before any search.
5. `app.py`: Streamlit screen (physician -> patient -> question -> cited answer + sources).

## Run (VS Code terminal in this folder)
1. Copy your `.env` from `patient-chart-rag` into this folder (keys: PINECONE_API_KEY, PINECONE_INDEX, NEBIUS_API_KEY, NEBIUS_MODEL).
2. `python -m venv .venv` then `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`
3. `.\.venv\Scripts\python.exe build_documents.py`
4. `.\.venv\Scripts\python.exe ingest.py --upload` (about 5 to 15 minutes the first time)
5. `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8503`

## Rebuild the 200-patient data (only if needed)
1. `python make_200_patient_subset.py ..\patient-chart-rag\data\raw\synthea_csv_nov2021\csv data\raw\synthea_csv_nov2021\csv`
2. `python relabel_ids.py data\raw\synthea_csv_nov2021\csv` (patients ptA..., visits enaaa..., payers 101...; mapping in id_map.csv)
3. `python check_subset.py data\raw\synthea_csv_nov2021\csv`
Then run steps 3 and 4 above again.

## Known limits
- Care plans are labels only, such as "Diabetes self management plan", with a start date, an optional stop date
  (empty means still active) and a reason. Synthea does not provide the plan's detailed instructions, so the app
  can say which care plan is active and why, but not the step-by-step instructions given to the patient.
- No appointment scheduling data: the app cannot say when the next appointment is.
- "No allergies recorded" means none are documented, not that the patient has none.
- Old code from the 1,163-patient version is kept in `_old_version/` and is not used.
