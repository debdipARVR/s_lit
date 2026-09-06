"""Automated Slush-Pile Webhook Dispatcher with HMAC-SHA256 Signatures.

Delivers real-time manuscript screening verdicts to publishers' editorial submission systems
(Submittable, QueryTracker, Moksha, custom CMS/CRM).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

from ..queue.batch_models import BatchManuscriptJob

logger = logging.getLogger("SlushPileWebhook")


def compute_decision_recommendation(human_score: float) -> str:
    """Compute automated slush-pile triaging recommendation."""
    if human_score >= 85.0:
        return "AUTO_APPROVE_HUMAN_CADENCE"
    elif human_score >= 50.0:
        return "EDITOR_REVIEW_FLAGGED_AI_POLISH"
    else:
        return "REJECT_EXCESSIVE_SYNTHETIC_GENERATION"


def create_webhook_payload(job: BatchManuscriptJob, base_url: str = "https://manuscriptproof.com") -> Dict[str, Any]:
    """Generate structured publisher callback payload."""
    recommendation = compute_decision_recommendation(job.overall_human_score)
    
    return {
        "event": "manuscript.scan.completed",
        "job_id": job.job_id,
        "tenant_id": job.tenant_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manuscript": {
            "title": job.title,
            "author_name": job.author_name,
            "publisher_name": job.publisher_name,
            "total_words": job.total_words,
            "total_chapters": job.total_chapters,
        },
        "forensics": {
            "human_authenticity_score": round(job.overall_human_score, 2),
            "ai_probability": round(job.overall_ai_probability, 2),
            "verdict": job.overall_verdict,
            "decision_recommendation": recommendation,
            "attributed_foundation_model": job.top_model_provenance,
            "resonance_gap_percent": round(job.resonance_gap, 2),
            "openssh_digital_signature": job.openssh_signature,
        },
        "chapter_heatmap": [
            {
                "chapter_index": c.chapter_index,
                "title": c.title,
                "word_count": c.word_count,
                "human_authenticity_score": round(c.human_authenticity_score, 2),
                "ai_probability": round(c.ai_probability, 2),
                "verdict": c.verdict,
            }
            for c in job.chapter_results
        ],
        "certificate_url": f"{base_url}/api/v1/certificates/{job.job_id}.pdf",
    }


def sign_payload(payload_json: str, secret_key: str) -> str:
    """Compute HMAC-SHA256 signature for payload verification."""
    mac = hmac.new(secret_key.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def dispatch_manuscript_webhook(
    job: BatchManuscriptJob,
    callback_url: str,
    secret_key: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Dispatch HMAC-authenticated HTTP POST webhook to publisher callback URL."""
    if not callback_url:
        return {"status": "skipped", "reason": "No callback URL provided."}

    secret = secret_key or "manuscriptproof_default_webhook_secret_2026"
    payload = create_webhook_payload(job)
    payload_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
    sig_header = sign_payload(payload_json, secret)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ManuscriptProof-SlushPile-Webhook/1.0",
        "X-ManuscriptProof-Signature": sig_header,
        "X-ManuscriptProof-Job-Id": job.job_id,
        "X-ManuscriptProof-Event": "manuscript.scan.completed",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(callback_url, content=payload_json, headers=headers)
            return {
                "status": "delivered" if resp.is_success else "failed",
                "status_code": resp.status_code,
                "response_body": resp.text[:500],
                "job_id": job.job_id,
                "signature": sig_header,
            }
    except Exception as e:
        logger.warning(f"Failed to post webhook to {callback_url}: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "job_id": job.job_id,
            "signature": sig_header,
        }
