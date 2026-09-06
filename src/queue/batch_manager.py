"""Persistent SQLite Batch Manuscript Queue Manager for ManuscriptProof 1.0.

Provides ACID-compliant, encrypted-at-rest queuing for 100,000+ word novels.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..security.encryption import encrypt_at_rest, decrypt_at_rest
from .batch_models import BatchManuscriptJob, ChapterResult, JobStatus
from .chunker import split_manuscript_into_chapters

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "sessions.db"


class BatchManager:
    """Manages batch manuscript evaluation jobs in SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Initialize batch_jobs table."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author_name TEXT NOT NULL,
                    publisher_name TEXT NOT NULL,
                    total_words INTEGER NOT NULL,
                    total_chapters INTEGER NOT NULL,
                    processed_chapters INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    progress_pct REAL DEFAULT 0.0,
                    overall_human_score REAL DEFAULT 0.0,
                    overall_ai_probability REAL DEFAULT 0.0,
                    overall_verdict TEXT DEFAULT 'Pending Analysis',
                    top_model_provenance TEXT DEFAULT 'Undetermined',
                    resonance_gap REAL DEFAULT 0.0,
                    enc_manuscript_text TEXT,
                    enc_chapter_results_json TEXT,
                    error_message TEXT,
                    callback_url TEXT,
                    openssh_signature TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_tenant ON batch_jobs(tenant_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_jobs(status)")
            conn.commit()

    def submit_job(
        self,
        tenant_id: str,
        title: str,
        text: str,
        author_name: str = "Anonymous Author",
        publisher_name: str = "Penguin Random House",
        callback_url: Optional[str] = None,
    ) -> BatchManuscriptJob:
        """Partition manuscript into chapters and enqueue a new batch job."""
        chapters = split_manuscript_into_chapters(text)
        total_words = len(text.split()) if text else 0
        total_chapters = len(chapters)
        job_id = f"job_{uuid.uuid4().hex[:14]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        enc_text = encrypt_at_rest(text)
        empty_chapters_json = encrypt_at_rest(json.dumps([]))

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO batch_jobs (
                    job_id, tenant_id, title, author_name, publisher_name,
                    total_words, total_chapters, processed_chapters, status,
                    progress_pct, overall_human_score, overall_ai_probability,
                    overall_verdict, top_model_provenance, resonance_gap,
                    enc_manuscript_text, enc_chapter_results_json, callback_url,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0.0, 0.0, 0.0, 'Pending Analysis', 'Undetermined', 0.0, ?, ?, ?, ?, ?)
            """, (
                job_id, tenant_id, title, author_name, publisher_name,
                total_words, total_chapters, JobStatus.QUEUED.value,
                enc_text, empty_chapters_json, callback_url, now, now
            ))
            conn.commit()

        return BatchManuscriptJob(
            job_id=job_id,
            tenant_id=tenant_id,
            title=title,
            author_name=author_name,
            publisher_name=publisher_name,
            total_words=total_words,
            total_chapters=total_chapters,
            status=JobStatus.QUEUED,
            callback_url=callback_url,
            created_at=now,
        )

    def get_job(self, job_id: str) -> Optional[BatchManuscriptJob]:
        """Retrieve batch job by job_id."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            return self._row_to_job(dict(row)) if row else None

    def get_manuscript_text(self, job_id: str) -> Optional[str]:
        """Fetch and decrypt raw manuscript text for a job."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT enc_manuscript_text FROM batch_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row and row["enc_manuscript_text"]:
                return decrypt_at_rest(row["enc_manuscript_text"])
        return None

    def list_jobs(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[BatchManuscriptJob]:
        """List recent jobs with optional filtering."""
        query = "SELECT * FROM batch_jobs"
        params = []
        conditions = []

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_job(dict(r)) for r in rows]

    def claim_next_job(self) -> Optional[BatchManuscriptJob]:
        """Atomically claim next available queued job."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT job_id FROM batch_jobs 
                WHERE status = ? 
                ORDER BY created_at ASC LIMIT 1
            """, (JobStatus.QUEUED.value,))
            row = cursor.fetchone()
            if not row:
                return None

            job_id = row["job_id"]
            cursor.execute("""
                UPDATE batch_jobs 
                SET status = ?, updated_at = ? 
                WHERE job_id = ? AND status = ?
            """, (JobStatus.PROCESSING.value, now, job_id, JobStatus.QUEUED.value))
            conn.commit()

            if cursor.rowcount > 0:
                return self.get_job(job_id)
        return None

    def update_progress(
        self,
        job_id: str,
        processed_chapters: int,
        total_chapters: int,
        chapter_results: List[ChapterResult],
        current_human_score: float,
        current_ai_prob: float,
        verdict: str,
        top_model: str,
        resonance_gap: float,
    ) -> None:
        """Update job progress and partial chapter results in SQLite."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pct = (processed_chapters / max(1, total_chapters)) * 100.0
        chap_dicts = [c.__dict__ for c in chapter_results]
        enc_chapters = encrypt_at_rest(json.dumps(chap_dicts))

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE batch_jobs 
                SET processed_chapters = ?, progress_pct = ?,
                    overall_human_score = ?, overall_ai_probability = ?,
                    overall_verdict = ?, top_model_provenance = ?,
                    resonance_gap = ?, enc_chapter_results_json = ?,
                    updated_at = ?
                WHERE job_id = ?
            """, (
                processed_chapters, pct, current_human_score, current_ai_prob,
                verdict, top_model, resonance_gap, enc_chapters, now, job_id
            ))
            conn.commit()

    def complete_job(
        self,
        job_id: str,
        overall_human_score: float,
        overall_ai_prob: float,
        verdict: str,
        top_model: str,
        resonance_gap: float,
        chapter_results: List[ChapterResult],
        openssh_signature: str,
    ) -> None:
        """Mark job as completed and save final forensic results."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        chap_dicts = [c.__dict__ for c in chapter_results]
        enc_chapters = encrypt_at_rest(json.dumps(chap_dicts))

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE batch_jobs 
                SET status = ?, progress_pct = 100.0,
                    processed_chapters = total_chapters,
                    overall_human_score = ?, overall_ai_probability = ?,
                    overall_verdict = ?, top_model_provenance = ?,
                    resonance_gap = ?, enc_chapter_results_json = ?,
                    openssh_signature = ?, updated_at = ?, completed_at = ?
                WHERE job_id = ?
            """, (
                JobStatus.COMPLETED.value, overall_human_score, overall_ai_prob,
                verdict, top_model, resonance_gap, enc_chapters,
                openssh_signature, now, now, job_id
            ))
            conn.commit()

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed with error details."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE batch_jobs 
                SET status = ?, error_message = ?, updated_at = ?, completed_at = ?
                WHERE job_id = ?
            """, (JobStatus.FAILED.value, error_message, now, now, job_id))
            conn.commit()

    def _row_to_job(self, row: Dict[str, Any]) -> BatchManuscriptJob:
        """Convert database row to BatchManuscriptJob."""
        chapter_results = []
        if row.get("enc_chapter_results_json"):
            try:
                dec = decrypt_at_rest(row["enc_chapter_results_json"])
                parsed = json.loads(dec)
                chapter_results = [ChapterResult(**c) for c in parsed]
            except Exception:
                pass

        return BatchManuscriptJob(
            job_id=row["job_id"],
            tenant_id=row["tenant_id"],
            title=row["title"],
            author_name=row["author_name"],
            publisher_name=row["publisher_name"],
            total_words=row["total_words"],
            total_chapters=row["total_chapters"],
            processed_chapters=row["processed_chapters"],
            status=JobStatus(row["status"]),
            progress_pct=row["progress_pct"],
            overall_human_score=row["overall_human_score"],
            overall_ai_probability=row["overall_ai_probability"],
            overall_verdict=row["overall_verdict"],
            top_model_provenance=row["top_model_provenance"],
            resonance_gap=row["resonance_gap"],
            chapter_results=chapter_results,
            error_message=row["error_message"],
            callback_url=row["callback_url"],
            openssh_signature=row["openssh_signature"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )


# Global singleton
_batch_manager: Optional[BatchManager] = None

def get_batch_manager() -> BatchManager:
    """Retrieve global BatchManager singleton."""
    global _batch_manager
    if _batch_manager is None:
        _batch_manager = BatchManager()
    return _batch_manager
