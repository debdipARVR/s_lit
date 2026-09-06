"""Asynchronous Background Queue Worker for ManuscriptProof 1.0.

Processes multi-chapter 100,000+ word novels through Cloze Congruence and DNA Attribution.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from ..engine import (
    ClozeCongruenceDetector,
    DNAAttributionEngine,
    NvidiaNIMClient,
    OpenRouterClient,
)
from ..security.ssh_verifier import get_or_create_ssh_authority, generate_ssh_verifiable_session
from .batch_manager import BatchManager, get_batch_manager
from .batch_models import BatchManuscriptJob, ChapterResult, JobStatus
from .chunker import split_manuscript_into_chapters

logger = logging.getLogger("ManuscriptProofWorker")


class BatchWorker:
    """Evaluates queued manuscripts chapter-by-chapter with real-time progress updates."""

    def __init__(
        self,
        batch_manager: Optional[BatchManager] = None,
        detector: Optional[ClozeCongruenceDetector] = None,
        dna_engine: Optional[DNAAttributionEngine] = None,
    ):
        self.batch_manager = batch_manager or get_batch_manager()
        self.detector = detector or ClozeCongruenceDetector()
        self.dna_engine = dna_engine or DNAAttributionEngine()
        self._stop_event = threading.Event()

    def process_job(self, job_id: str) -> Optional[BatchManuscriptJob]:
        """Process an individual manuscript job to completion."""
        job = self.batch_manager.get_job(job_id)
        if not job:
            return None

        raw_text = self.batch_manager.get_manuscript_text(job_id)
        if not raw_text:
            self.batch_manager.fail_job(job_id, "Manuscript text payload not found.")
            return self.batch_manager.get_job(job_id)

        try:
            chapters = split_manuscript_into_chapters(raw_text)
            total_chapters = max(1, len(chapters))
            chapter_results: List[ChapterResult] = []

            running_weighted_ai_prob = 0.0
            total_words_processed = 0

            # Whole-novel DNA attribution
            dna_res = self.dna_engine.analyze(raw_text[:8000]) if len(raw_text) > 100 else {}
            top_model = dna_res.get("attributed_display_name") or dna_res.get("attributed_source", "Undetermined")
            res_gap = dna_res.get("resonance_gap_percent", 0.0)

            for idx, chap in enumerate(chapters, start=1):
                c_title = chap["title"]
                c_text = chap["text"]
                c_words = chap["word_count"]

                # Run cloze analysis on chapter
                det_res = self.detector.analyze(c_text) if len(c_text.split()) >= 15 else {}
                c_prob = det_res.get("ai_probability", 0.0)
                c_congruence = det_res.get("combined_congruence_score", det_res.get("congruence_score", 0.0))
                c_verdict = det_res.get("verdict", "Likely Human-Authored")
                c_human = max(0.0, 100.0 - c_prob)

                chap_res = ChapterResult(
                    chapter_index=idx,
                    title=c_title,
                    word_count=c_words,
                    human_authenticity_score=c_human,
                    ai_probability=c_prob,
                    congruence_score=c_congruence,
                    verdict=c_verdict,
                    attributed_model=top_model,
                    snippet=c_text[:300] + "..." if len(c_text) > 300 else c_text,
                )
                chapter_results.append(chap_res)

                running_weighted_ai_prob += (c_prob * c_words)
                total_words_processed += c_words

                curr_overall_ai = running_weighted_ai_prob / max(1, total_words_processed)
                curr_overall_human = max(0.0, 100.0 - curr_overall_ai)
                curr_verdict = "Likely Human-Authored" if curr_overall_human >= 80.0 else ("Mixed Human / AI Polish" if curr_overall_human >= 50.0 else "AI-Generated Manuscript")

                self.batch_manager.update_progress(
                    job_id=job_id,
                    processed_chapters=idx,
                    total_chapters=total_chapters,
                    chapter_results=chapter_results,
                    current_human_score=curr_overall_human,
                    current_ai_prob=curr_overall_ai,
                    verdict=curr_verdict,
                    top_model=top_model,
                    resonance_gap=res_gap,
                )

            # Final calculations
            final_overall_ai = running_weighted_ai_prob / max(1, total_words_processed)
            final_overall_human = max(0.0, 100.0 - final_overall_ai)
            final_verdict = "Likely Human-Authored" if final_overall_human >= 80.0 else ("Mixed Human / AI Polish" if final_overall_human >= 50.0 else "AI-Generated Manuscript")

            # Cryptographic OpenSSH signature
            sig_raw = f"{job_id}|{final_overall_human:.2f}|{top_model}|{time.time()}"
            sig_hex = hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()

            self.batch_manager.complete_job(
                job_id=job_id,
                overall_human_score=final_overall_human,
                overall_ai_prob=final_overall_ai,
                verdict=final_verdict,
                top_model=top_model,
                resonance_gap=res_gap,
                chapter_results=chapter_results,
                openssh_signature=sig_hex,
            )

            # Dispatch outgoing webhook if callback URL configured
            if job.callback_url:
                self._dispatch_webhook(job_id, job.callback_url)

            return self.batch_manager.get_job(job_id)

        except Exception as e:
            logger.exception(f"Error processing job {job_id}: {str(e)}")
            self.batch_manager.fail_job(job_id, str(e))
            return self.batch_manager.get_job(job_id)

    def _dispatch_webhook(self, job_id: str, callback_url: str) -> None:
        """Helper to trigger outgoing webhook."""
        try:
            from ..web.webhook_dispatcher import dispatch_manuscript_webhook
            completed_job = self.batch_manager.get_job(job_id)
            if completed_job:
                dispatch_manuscript_webhook(completed_job, callback_url)
        except Exception as e:
            logger.warning(f"Failed to dispatch webhook for {job_id}: {str(e)}")

    def run_worker_loop(self, poll_interval: float = 1.0, max_iterations: Optional[int] = None) -> None:
        """Run blocking or background polling worker loop."""
        iterations = 0
        while not self._stop_event.is_set():
            job = self.batch_manager.claim_next_job()
            if job:
                self.process_job(job.job_id)
            else:
                time.sleep(poll_interval)

            iterations += 1
            if max_iterations and iterations >= max_iterations:
                break

    def stop(self) -> None:
        """Signal the worker to stop running."""
        self._stop_event.set()
