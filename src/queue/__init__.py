"""ManuscriptProof 1.0 Batch Manuscript Queue Subsystem."""

from .batch_models import BatchManuscriptJob, ChapterResult, JobStatus
from .batch_manager import BatchManager, get_batch_manager
from .chunker import split_manuscript_into_chapters, split_manuscript_into_pages
from .worker import BatchWorker

__all__ = [
    "BatchManuscriptJob",
    "ChapterResult",
    "JobStatus",
    "BatchManager",
    "get_batch_manager",
    "split_manuscript_into_chapters",
    "split_manuscript_into_pages",
    "BatchWorker",
]
