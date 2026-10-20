"""Decision store — Cairn-style local libSQL: text + F32_BLOB embeddings.

One local DB file holds each decision's text and its meaning-vector side by side.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from embeddings import Embedder, get_embedder

ROOT = Path(__file__).parent
DEFAULT_DB_PATH = ROOT / "data" / "decisions_vec.db"


def _vector_literal(vec: list[float]) -> str:
    """
    Turn a Python list of floats into the string libSQL's vector32() needs,
    e.g. '[0.1,0.2,0.3]'.
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class DecisionStore:
    """
    Local database for decisions.
    Creates/opens decisions_vec.db, stores text + embedding per row,
    and can load everything back for duplicate finding.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Open (or create) the DB file, load an embedder, create tables if needed.
        This does NOT fill data yet — ingest_json() does that.
        """
        try:
            import libsql_experimental as libsql
        except ImportError as err:
            raise RuntimeError(
                'libSQL not installed. Run: pip install "libsql-experimental"'
            ) from err

        self.db_path = str(
            db_path
            or os.environ.get("DECISIONS_DB_PATH")
            or DEFAULT_DB_PATH
        )
        Path(self.db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()
        self.dim = self.embedder.dim
        self._conn = libsql.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """
        Create tables/index if they do not exist:
        - decisions: id, decision_id, content (text), metadata, embedding (vector)
        - store_meta: small key/value notes (e.g. which JSON file we ingested)
        Also creates a libSQL vector index on embedding for nearest-neighbor lookup.
        """
        # INTEGER PK + F32_BLOB matches Cairn's memory_entries shape;
        # libSQL's vector index keys off rowid.
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS decisions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT NOT NULL UNIQUE,
                content    TEXT NOT NULL,
                metadata   TEXT,
                created_at INTEGER NOT NULL DEFAULT (unixepoch()),
                embedding  F32_BLOB({self.dim})
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS decisions_vec_idx "
            "ON decisions(libsql_vector_idx(embedding))"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS store_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def count(self) -> int:
        """How many decision rows are currently in the DB."""
        row = self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return int(row[0]) if row else 0

    def get_meta(self, key: str) -> str | None:
        """Read one value from store_meta (e.g. source_key)."""
        row = self._conn.execute(
            "SELECT value FROM store_meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        """Write/update one value in store_meta."""
        self._conn.execute(
            "INSERT INTO store_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Delete all decisions and meta rows (used before a full re-ingest)."""
        self._conn.execute("DELETE FROM decisions")
        self._conn.execute("DELETE FROM store_meta")
        self._conn.commit()

    def ingest_json(
        self,
        path: Path,
        *,
        batch_size: int = 64,
        force: bool = False,
    ) -> int:
        """
        Fill the DB from a JSON file.
        Steps:
        1) Read JSON array of {id, text, ...}
        2) If DB already matches this file and force=False → skip (fast path)
        3) Else clear DB, embed texts in batches, INSERT text + vector per row
        Returns the number of rows stored.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of decision objects")

        source_key = f"{path.resolve()}::{path.stat().st_mtime_ns}::{len(data)}"
        if not force and self.get_meta("source_key") == source_key and self.count() == len(data):
            print(f"DB already up to date ({self.count()} rows) → {self.db_path}")
            return self.count()

        if force or self.count() > 0:
            self.clear()

        print(
            f"Ingesting {len(data)} decisions into {self.db_path} "
            f"(embedder={self.embedder.name}, dim={self.dim}) …"
        )
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            texts = [str(row["text"]) for row in batch]
            vectors = self.embedder.embed(texts)
            for row, vec in zip(batch, vectors):
                meta = {
                    k: v for k, v in row.items() if k not in {"id", "text"}
                }
                self._conn.execute(
                    """
                    INSERT INTO decisions
                        (decision_id, content, metadata, embedding)
                    VALUES (?, ?, ?, vector32(?))
                    """,
                    (
                        str(row.get("id", f"row_{start}")),
                        str(row["text"]),
                        json.dumps(meta, ensure_ascii=False) if meta else None,
                        _vector_literal(vec),
                    ),
                )
            self._conn.commit()
            done = min(start + batch_size, len(data))
            print(f"  {done}/{len(data)}")

        self.set_meta("source_key", source_key)
        self.set_meta("embedder", self.embedder.name)
        self.set_meta("dim", str(self.dim))
        return self.count()

    def load_all(self) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
        """
        Read every decision from the DB for clustering.
        Returns:
        - decisions: list of dicts with id/text/(extra metadata)
        - texts: list of content strings in the same order
        - embedding matrix: numpy array shape (N, dim), L2-normalized
        """
        rows = self._conn.execute(
            """
            SELECT decision_id, content, metadata,
                   vector_extract(embedding)
            FROM decisions
            ORDER BY id
            """
        ).fetchall()
        decisions: list[dict[str, Any]] = []
        texts: list[str] = []
        vectors: list[list[float]] = []
        for decision_id, content, metadata, emb in rows:
            meta = json.loads(metadata) if metadata else {}
            decisions.append(
                {
                    "id": decision_id,
                    "text": content,
                    **meta,
                }
            )
            texts.append(content)
            # vector_extract returns a string like "[0.1,0.2,...]" or a list
            if isinstance(emb, str):
                vec = json.loads(emb)
            else:
                vec = list(emb)
            vectors.append([float(x) for x in vec])
        emb_mat = np.asarray(vectors, dtype=np.float32)
        # Ensure unit length for cosine via dot product
        norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        emb_mat = emb_mat / norms
        return decisions, texts, emb_mat

    def nearest(self, text: str, k: int = 10) -> list[tuple[str, str, float]]:
        """
        Ask the DB: which stored decisions are closest in meaning to this text?
        Uses libSQL vector_top_k (optional helper; main clustering uses sklearn).
        Returns list of (decision_id, content, cosine_distance).
        """
        vec = _vector_literal(self.embedder.embed_one(text))
        rows = self._conn.execute(
            """
            SELECT d.decision_id, d.content,
                   vector_distance_cos(d.embedding, vector32(?)) AS dist
            FROM vector_top_k('decisions_vec_idx', vector32(?), ?) AS v
            JOIN decisions d ON d.rowid = v.key
            ORDER BY dist
            """,
            (vec, vec, k),
        ).fetchall()
        return [(str(r[0]), str(r[1]), float(r[2])) for r in rows]
