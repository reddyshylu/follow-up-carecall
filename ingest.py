"""Step 2: chunk the documents, embed them, and load them into Pinecone with metadata.

Usage:
    python ingest.py            # build chunks only (no keys needed) and print counts
    python ingest.py --upload   # also embed with BGE and upload to Pinecone (needs .env)

Follows the course pipeline: Documents with metadata -> RecursiveCharacterTextSplitter
-> embeddings -> Pinecone with metadata, so search can be filtered to one patient.
Each chunk starts with a short header (patient, document title) so it still makes
sense when read on its own.
"""
print("Step 2: loading libraries (this can take 1 to 2 minutes with no output, please wait)...", flush=True)
import argparse
import hashlib
import json
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import (CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS, DOCUMENTS, EMBED_MODEL, MANIFEST, NAMESPACE,
                    AppError, embedder, read_jsonl, settings, write_jsonl)

parser = argparse.ArgumentParser()
parser.add_argument("--upload", action="store_true", help="embed and upload to Pinecone")
parser.add_argument("--resume", action="store_true", help="continue an interrupted upload instead of starting over")
args = parser.parse_args()

if not DOCUMENTS.exists():
    raise SystemExit("Run build_documents.py first.")
raw = read_jsonl(DOCUMENTS)
docs = [Document(page_content=d["text"], metadata={**d["metadata"], "doc_id": d["doc_id"]}) for d in raw]
sources = {d["doc_id"]: d["sources"] for d in raw}

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
chunks = []
for doc in docs:
    m = doc.metadata
    header = f"[{m['title']} | patient {m['patient_id']} | physician {m['physician']}]\n"
    for i, piece in enumerate(splitter.split_text(doc.page_content)):
        chunks.append({"chunk_id": f"{m['doc_id']}#{i}", "text": header + piece, "metadata": {**m, "chunk": i},
                       "sources": sources[m["doc_id"]]})
write_jsonl(CHUNKS, chunks)
by_type = {}
for c in chunks:
    by_type[c["metadata"]["doc_type"]] = by_type.get(c["metadata"]["doc_type"], 0) + 1
print(f"{len(docs):,} documents -> {len(chunks):,} chunks ({', '.join(f'{k} {v:,}' for k, v in by_type.items())})")
print(f"Saved {CHUNKS}")

fingerprint = hashlib.sha256(CHUNKS.read_bytes()).hexdigest()[:16]
manifest = {"namespace": NAMESPACE, "model": EMBED_MODEL, "chunks": len(chunks), "documents": len(docs),
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "chunks_sha256_16": fingerprint,
            "patients": len({c["metadata"]["patient_id"] for c in chunks}), "uploaded": False}

# Keep the "uploaded" status when the chunks are unchanged and we are not uploading again.
if not args.upload and MANIFEST.exists():
    previous = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if previous.get("uploaded") and previous.get("chunks_sha256_16") == fingerprint:
        manifest = previous

if args.upload:
    config = settings()
    if not config["PINECONE_API_KEY"] or not config["PINECONE_INDEX"]:
        raise AppError("Add PINECONE_API_KEY and PINECONE_INDEX to .env first.")
    from pinecone import Pinecone
    pc = Pinecone(api_key=config["PINECONE_API_KEY"])
    info = pc.describe_index(config["PINECONE_INDEX"])
    if info.dimension != 384:
        raise AppError(f"Index {config['PINECONE_INDEX']} has dimension {info.dimension}; BGE-small needs 384.")
    index = pc.Index(config["PINECONE_INDEX"])
    # Start clean so no stale chunks from an earlier run stay behind. Only CareCall's own namespace is cleared.
    progress_file = MANIFEST.parent / "upload_progress.txt"
    done = int(progress_file.read_text()) if args.resume and progress_file.exists() else 0
    if done:
        print(f"Resuming after {done:,} chunks already uploaded.", flush=True)
    else:
        try:
            index.delete(delete_all=True, namespace=NAMESPACE)
        except Exception:
            pass  # namespace did not exist yet
    print("Loading the BGE embedding model (first time downloads about 130 MB)...", flush=True)
    model = embedder()
    print("Embedding and uploading chunks...", flush=True)
    start = time.time()
    batch = 96
    for s in range(done, len(chunks), batch):
        part = chunks[s:s + batch]
        vectors = model.encode([c["text"] for c in part], normalize_embeddings=True, batch_size=32)
        payload = [
            {"id": c["chunk_id"], "values": v.tolist(),
             "metadata": {k: c["metadata"][k] for k in ("patient_id", "physician", "doc_type", "encounter_id",
                                                        "date", "encounter_class", "title", "doc_id", "chunk")}}
            for c, v in zip(part, vectors)]
        for attempt in range(5):  # survive short network drops (for example, switching Wi-Fi)
            try:
                index.upsert(namespace=NAMESPACE, vectors=payload)
                break
            except Exception as error:
                if attempt == 4:
                    raise AppError(f"Upload stopped at chunk {s:,} ({type(error).__name__}). "
                                   "Run: python ingest.py --upload --resume") from None
                print(f"  network problem, retrying in {10 * (attempt + 1)} s...", flush=True)
                time.sleep(10 * (attempt + 1))
        progress_file.write_text(str(s + len(part)))
        if (s // batch) % 5 == 0:
            print(f"  uploaded {min(s + batch, len(chunks)):,}/{len(chunks):,} ({time.time() - start:.0f}s)", flush=True)
    # Pinecone is eventually consistent: wait until the namespace reports every vector.
    for _ in range(30):
        count = index.describe_index_stats().namespaces.get(NAMESPACE)
        count = getattr(count, "vector_count", 0) if count else 0
        if count >= len(chunks):
            break
        time.sleep(2)
    print(f"Pinecone namespace {NAMESPACE}: {count:,} vectors (expected {len(chunks):,})")
    if count < len(chunks):
        raise AppError("Not every vector is visible yet. Wait a minute and run: python ingest.py --upload again.")
    progress_file.unlink(missing_ok=True)
    manifest.update({"uploaded": True, "index": config["PINECONE_INDEX"], "uploaded_at": time.strftime("%Y-%m-%d %H:%M")})

MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Manifest: {json.dumps(manifest)}")
