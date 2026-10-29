"""Shared settings and helpers for CareCall (Week 2 RAG, 200 synthetic patients)."""
import json
import os
from pathlib import Path

# Do not write TLS session secrets to an inherited logging path; use the Windows certificate store.
os.environ.pop("SSLKEYLOGFILE", None)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data/processed"
DOCUMENTS = PROCESSED / "documents.jsonl"
CHUNKS = PROCESSED / "chunks.jsonl"
MANIFEST = PROCESSED / "carecall_manifest.json"

APP_NAME = "Follow-Up CareCall"
NAMESPACE = "carecall-200-v1"            # new Pinecone namespace; the old project's data is not touched
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 dimensions, runs locally, same model as before
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for BGE queries
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 150    # characters; well under BGE's 512-token limit
NEBIUS_BASE = "https://api.tokenfactory.nebius.com/v1/"


class AppError(Exception):
    """A message that is safe to show in the app (never contains keys or provider response bodies)."""


def settings():
    load_dotenv(ROOT / ".env", override=True)
    return {k: os.getenv(k, "").strip() for k in ("PINECONE_API_KEY", "PINECONE_INDEX", "NEBIUS_API_KEY", "NEBIUS_MODEL")}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


_model = None


def embedder():
    """Load the BGE model once (first load can take ~30 seconds)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL, cache_folder=str(ROOT / "data/model_cache"))
    return _model
