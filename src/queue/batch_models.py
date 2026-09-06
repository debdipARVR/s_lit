"""Batch Manuscript Queue Models and Data Structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ChapterResult:
    chapter_index: int
    title: str
    word_count: int
    human_authenticity_score: float = 100.0
    ai_probability: float = 0.0
    congruence_score: float = 0.0
    verdict: str = "Likely Human-Authored"
    attributed_model: str = "Human"
    snippet: str = ""


@dataclass
class BatchManuscriptJob:
    job_id: str
    tenant_id: str
    title: str
    author_name: str
    publisher_name: str
    total_words: int
    total_chapters: int
    processed_chapters: int = 0
    status: JobStatus = JobStatus.QUEUED
    progress_pct: float = 0.0
    overall_human_score: float = 0.0
    overall_ai_probability: float = 0.0
    overall_verdict: str = "Pending Analysis"
    top_model_provenance: str = "Undetermined"
    resonance_gap: float = 0.0
    chapter_results: List[ChapterResult] = field(default_factory=list)
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    openssh_signature: Optional[str] = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to JSON-serializable dictionary."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "author_name": self.author_name,
            "publisher_name": self.publisher_name,
            "total_words": self.total_words,
            "total_chapters": self.total_chapters,
            "processed_chapters": self.processed_chapters,
            "status": self.status.value,
            "progress_pct": round(self.progress_pct, 1),
            "overall_human_score": round(self.overall_human_score, 2),
            "overall_ai_probability": round(self.overall_ai_probability, 2),
            "overall_verdict": self.overall_verdict,
            "top_model_provenance": self.top_model_provenance,
            "resonance_gap": round(self.resonance_gap, 2),
            "chapter_results": [
                {
                    "chapter_index": c.chapter_index,
                    "title": c.title,
                    "word_count": c.word_count,
                    "human_authenticity_score": round(c.human_authenticity_score, 2),
                    "ai_probability": round(c.ai_probability, 2),
                    "congruence_score": round(c.congruence_score, 2),
                    "verdict": c.verdict,
                    "attributed_model": c.attributed_model,
                    "snippet": c.snippet,
                }
                for c in self.chapter_results
            ],
            "error_message": self.error_message,
            "callback_url": self.callback_url,
            "openssh_signature": self.openssh_signature,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
