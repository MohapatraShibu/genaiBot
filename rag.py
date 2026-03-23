# RAG pipeline load docs -> chunk -> embed -> store in sqlite-vec -> retrieve -> generate

import os
import re
import sqlite3
import struct
import hashlib
import logging
from pathlib import Path
from functools import lru_cache

import sqlite_vec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", 3))
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
DB_PATH = "rag_store.db"
CHUNK_SIZE = 300  # characters
CHUNK_OVERLAP = 50


# embedding model (loaded once)
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


def embed(text: str) -> list[float]:
    return get_embedder().encode(text, normalize_embeddings=True).tolist()


def serialize(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# SQLite-vec store
def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def init_db():
    dim = len(embed("warmup"))
    db = get_db()
    db.executescript(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY,
            source  TEXT,
            content TEXT,
            hash    TEXT UNIQUE
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}]
        );
    """)
    db.commit()
    db.close()


# document loading & chunking
def chunk_text(text: str) -> list[str]:
    # split text into overlapping character chunks
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 30]


def load_docs(docs_dir: str = DOCS_DIR):
    # load all .md/.txt files and index new chunks
    db = get_db()
    indexed = 0
    for path in Path(docs_dir).glob("**/*"):
        if path.suffix not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        # strip markdown headers for cleaner chunks but keep content
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        for chunk in chunk_text(text):
            h = hashlib.md5(chunk.encode()).hexdigest()
            exists = db.execute("SELECT 1 FROM chunks WHERE hash=?", (h,)).fetchone()
            if exists:
                continue
            cur = db.execute(
                "INSERT INTO chunks (source, content, hash) VALUES (?,?,?)",
                (path.name, chunk, h),
            )
            vec = serialize(embed(chunk))
            db.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?,?)",
                (cur.lastrowid, vec),
            )
            indexed += 1
    db.commit()
    db.close()
    logger.info("Indexed %d new chunks from %s", indexed, docs_dir)


# retrieval
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    # return top-k chunks most similar to query
    q_vec = serialize(embed(query))
    db = get_db()
    rows = db.execute(
        """
        SELECT c.source, c.content, v.distance
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.chunk_id
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
        """,
        (q_vec, top_k),
    ).fetchall()
    db.close()
    return [{"source": r[0], "content": r[1], "score": r[2]} for r in rows]


# LLM call
def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    return (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def call_llm(prompt: str) -> str:
    import ollama as _ollama
    resp = _ollama.chat(
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 512},
    )
    return resp["message"]["content"].strip()


# query cache
_query_cache: dict[str, str] = {}


def answer(query: str) -> tuple[str, list[dict]]:
    # main entry: retrieve + generate
    # returns (answer_text, source_chunks)
    cache_key = query.lower().strip()
    if cache_key in _query_cache:
        logger.info("Cache hit for query: %s", query)
        return _query_cache[cache_key], []

    chunks = retrieve(query)
    if not chunks:
        return "I couldn't find relevant information in the knowledge base.", []

    prompt = build_prompt(query, chunks)
    result = call_llm(prompt)
    _query_cache[cache_key] = result
    return result, chunks
