"""FastAPI Web Application and Production Commercial REST API for ManuscriptProof 1.0.

Provides comprehensive enterprise endpoints for:
1. Multi-Tenant User Management & Stripe Billing (Author Pro $19.99/mo, Publisher B2B $499/mo).
2. Cloud Batch Manuscript Queue (100k+ word novel async analysis & chapter heatmaps).
3. Automated Slush-Pile Ingestion & Webhook API (Submittable/QueryTracker integration).
4. Co-Branded Cryptographic Authorship Certificates (ReportLab PDF).
5. Multi-Scale 4-Pass Cloze Infilling & 12D LLM DNA Fingerprinting.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..billing import (
    PlanTier,
    SubscriptionStatus,
    StripeManager,
    get_stripe_manager,
    CustomBrandingConfig,
)
from ..engine import (
    ClozeCongruenceDetector,
    DNAAttributionEngine,
    NvidiaNIMClient,
    NVIDIA_MODELS,
    OpenRouterClient,
    OPENROUTER_MODELS,
    DEFAULT_OPENROUTER_MODEL,
    TextHumanizer,
    HUMANIZER_MODES,
)
from ..engine.pdf_generator import generate_branded_authorship_certificate_pdf
from ..queue import (
    BatchManager,
    BatchWorker,
    JobStatus,
    get_batch_manager,
)
from ..security.encryption import (
    EncryptionError,
    decrypt_api_key,
    encrypt_api_key,
    generate_fernet_key,
    mask_api_key,
)
from .webhook_dispatcher import dispatch_manuscript_webhook

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="ManuscriptProof 1.0 - Commercial Book AI & Provenance Scanner API",
    description="Enterprise Black-Box Multi-Scale Cloze Infilling Detector, Model Fingerprinting & Slush-Pile Webhook API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Request & Response Schemas
# =========================================================================
class DetectRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Input text to evaluate")
    mask_rate: float = Field(0.30, ge=0.10, le=0.50, description="Cloze masking percentage")
    num_passes: int = Field(2, ge=1, le=5, description="Monte Carlo passes count")
    model_name: Optional[str] = Field("z-ai/glm-5.2", description="Infill model")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Infill temperature")
    raw_api_key: Optional[str] = Field(None, description="Optional plaintext API key")
    encrypted_token: Optional[str] = Field(None, description="Encrypted Fernet token")
    fernet_key: Optional[str] = Field(None, description="Fernet Secret key")


class Detect4PassRequest(BaseModel):
    text: str = Field(..., min_length=15, description="Input text to evaluate")
    pass1_trials: int = Field(2, ge=1, le=5, description="Number of Pass 1 isolated trials")
    pass2_trials: int = Field(3, ge=1, le=5, description="Number of Pass 2 dual void trials")
    pass3_trials: int = Field(2, ge=1, le=5, description="Number of Pass 3 centroid block trials")
    pass4_trials: int = Field(2, ge=1, le=5, description="Number of Pass 4 boundary anchor trials")
    model_name: Optional[str] = Field("nvidia/nemotron-3.5-lightning:free", description="Infilling foundation model")
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Greedy decoding temperature")
    api_key: Optional[str] = Field(None, description="Optional BYOK OpenRouter API key")


class DNAAttributionRequest(BaseModel):
    text: str = Field(..., min_length=25, description="Candidate synthetic text to attribute")
    candidate_models: Optional[List[str]] = Field(None, description="Subset of models to test against")
    api_key: Optional[str] = Field(None, description="Optional BYOK OpenRouter API key")


class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text to humanize")
    domain: str = Field("academic", description="Target domain profile")
    model_name: Optional[str] = Field("z-ai/glm-5.2", description="NVIDIA NIM model")
    temperature: float = Field(0.75, ge=0.1, le=1.2, description="Generation temperature")
    raw_api_key: Optional[str] = Field(None, description="Optional plaintext API key")
    encrypted_token: Optional[str] = Field(None, description="Encrypted Fernet token")
    fernet_key: Optional[str] = Field(None, description="Fernet Secret key")


class EncryptRequest(BaseModel):
    api_key: str = Field(..., min_length=1, description="Raw API key to encrypt")
    fernet_key: Optional[str] = Field(None, description="Optional custom Fernet secret key")


class PromptGenerateRequest(BaseModel):
    domain: str = Field("academic", description="Domain profile")
    target_audience: str = Field("General Audience", description="Target readership")
    additional_notes: str = Field("", description="Custom stylistic instructions")


# Commercial Slush-Pile & Batch Schemas
class SlushPileSubmitRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Manuscript title")
    text: str = Field(..., min_length=50, description="Full novel manuscript text")
    author_name: str = Field("Anonymous Author", description="Author full name")
    publisher_name: Optional[str] = Field("Penguin Random House Editorial", description="Publisher / Agency")
    callback_url: Optional[str] = Field(None, description="Webhook callback URL for scan completion")
    tenant_id: Optional[str] = Field(None, description="Optional tenant ID (if authenticated)")
    api_key: Optional[str] = Field(None, description="Publisher API key (starts with mp_live_)")


class CreateCheckoutRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    target_tier: str = Field("author_pro", description="author_pro or publisher_b2b")
    success_url: Optional[str] = Field("https://manuscriptproof.com/billing/success")
    cancel_url: Optional[str] = Field("https://manuscriptproof.com/billing/cancel")


class CreateApiKeyRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    key_name: str = Field("Slush-Pile Integration", description="Name/label for API key")


class UpdateBrandingRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    publisher_name: str = Field("Penguin Random House")
    brand_color_hex: str = Field("#0f172a")
    accent_color_hex: str = Field("#4f46e5")
    default_signatory_name: str = Field("Senior Acquisitions Editor")
    default_signatory_title: str = Field("Manuscript Review Board")


# =========================================================================
# Background Task Helper for Queue Worker
# =========================================================================
def run_batch_job_in_background(job_id: str):
    """Execute queue worker in a background thread."""
    worker = BatchWorker()
    worker.process_job(job_id)


# =========================================================================
# System Health & Status
# =========================================================================
@app.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """Health check for Docker / Kubernetes container orchestrators."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "service": "manuscriptproof-engine",
    }


@app.get("/api/status")
async def get_system_status() -> Dict[str, Any]:
    """Retrieve system configuration and commercial license status."""
    nim_status = NvidiaNIMClient().get_status()
    return {
        "status": "online",
        "service": "ManuscriptProof 1.0 Commercial AI & Provenance Scanner",
        "version": "1.0.0",
        "nvidia_client": nim_status,
        "available_modes": list(HUMANIZER_MODES.keys()),
        "tiers_supported": ["Free Author Trial", "Author Pro ($19.99/mo)", "Publisher B2B Enterprise ($499/mo)"],
        "max_manuscript_words": "Unlimited (Batch Queue)",
        "security": {
            "encryption_at_rest": "AES-256 (Fernet)",
            "verifiable_sessions": "OpenSSH Ed25519",
            "webhook_auth": "HMAC-SHA256 (RFC 2104)",
        },
        "available_models": NVIDIA_MODELS,
    }


@app.get("/api/models")
async def get_models() -> List[Dict[str, Any]]:
    """List supported prober models."""
    return NVIDIA_MODELS


# =========================================================================
# Commercial Slush-Pile & Batch Manuscript Queue Endpoints
# =========================================================================
@app.post("/api/v1/slushpile/submit")
async def submit_slushpile_manuscript(
    req: SlushPileSubmitRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Dict[str, Any]:
    """Submit a full book manuscript to the asynchronous slush-pile batch queue."""
    stripe_mgr = get_stripe_manager()
    batch_mgr = get_batch_manager()

    raw_key = req.api_key or x_api_key
    tenant_id = req.tenant_id

    # If API key provided, authenticate tenant
    if raw_key:
        tenant = stripe_mgr.verify_publisher_api_key(raw_key)
        if not tenant:
            raise HTTPException(status_code=401, detail="Invalid or deactivated Publisher API key.")
        tenant_id = tenant.tenant_id
        publisher_name = tenant.branding.publisher_name or req.publisher_name
    elif tenant_id:
        tenant = stripe_mgr.get_tenant_by_id(tenant_id)
        if not tenant:
            tenant = stripe_mgr.get_or_create_tenant(owner_email="author@example.com", organization_name="Author Workspace")
            tenant_id = tenant.tenant_id
        publisher_name = req.publisher_name
    else:
        # Default public sandbox tenant
        default_tenant = stripe_mgr.get_or_create_tenant(owner_email="guest@manuscriptproof.com", organization_name="Guest Publisher")
        tenant_id = default_tenant.tenant_id
        publisher_name = req.publisher_name

    # Check word quota
    word_count = len(req.text.split())
    within_quota, remaining = stripe_mgr.record_usage(tenant_id, word_count)
    if not within_quota:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly word quota exceeded for this account. Remaining quota: {remaining:,} words. Upgrade to Author Pro or Publisher B2B."
        )

    # Submit job
    job = batch_mgr.submit_job(
        tenant_id=tenant_id,
        title=req.title,
        text=req.text,
        author_name=req.author_name,
        publisher_name=publisher_name,
        callback_url=req.callback_url,
    )

    # Dispatch to background task runner
    background_tasks.add_task(run_batch_job_in_background, job.job_id)

    return {
        "status": "queued",
        "job_id": job.job_id,
        "title": job.title,
        "total_words": job.total_words,
        "total_chapters": job.total_chapters,
        "poll_status_url": f"/api/v1/slushpile/jobs/{job.job_id}",
        "certificate_url": f"/api/v1/certificates/{job.job_id}.pdf",
    }


@app.get("/api/v1/slushpile/jobs/{job_id}")
async def get_slushpile_job_status(job_id: str) -> Dict[str, Any]:
    """Query status, progress, and chapter-by-chapter heatmap of a batch manuscript job."""
    batch_mgr = get_batch_manager()
    job = batch_mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return job.to_dict()


@app.post("/api/v1/slushpile/webhook")
async def inbound_slushpile_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Inbound webhook receiver for Submittable, QueryTracker, or custom CMS submission portals."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    title = body.get("title") or body.get("submission_title") or "Untitled Submission"
    text = body.get("text") or body.get("manuscript_text") or body.get("content") or ""
    author = body.get("author") or body.get("author_name") or body.get("submitter_name") or "Anonymous Author"
    publisher = body.get("publisher") or body.get("organization") or "Slush-Pile Reviewer"
    callback_url = body.get("callback_url") or body.get("webhook_url")

    if not text or len(text.split()) < 15:
        raise HTTPException(status_code=400, detail="Manuscript text is missing or too short (< 15 words).")

    stripe_mgr = get_stripe_manager()
    default_tenant = stripe_mgr.get_or_create_tenant(owner_email="submissions@publisher.com", organization_name=publisher)

    batch_mgr = get_batch_manager()
    job = batch_mgr.submit_job(
        tenant_id=default_tenant.tenant_id,
        title=title,
        text=text,
        author_name=author,
        publisher_name=publisher,
        callback_url=callback_url,
    )

    background_tasks.add_task(run_batch_job_in_background, job.job_id)

    return {
        "status": "received_and_queued",
        "job_id": job.job_id,
        "title": title,
        "total_words": job.total_words,
        "total_chapters": job.total_chapters,
        "poll_url": f"/api/v1/slushpile/jobs/{job.job_id}",
    }


@app.get("/api/v1/certificates/{job_id}.pdf")
async def download_job_certificate(job_id: str) -> Response:
    """Download ReportLab PDF Certificate of Authorship for a completed batch job."""
    batch_mgr = get_batch_manager()
    job = batch_mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    stripe_mgr = get_stripe_manager()
    tenant = stripe_mgr.get_tenant_by_id(job.tenant_id)
    branding = tenant.branding if tenant else CustomBrandingConfig(publisher_name=job.publisher_name)

    chapter_dicts = [
        {
            "title": c.title,
            "word_count": c.word_count,
            "human_score": c.human_authenticity_score,
            "ai_probability": c.ai_probability,
            "verdict": c.verdict,
        }
        for c in job.chapter_results
    ]

    pdf_bytes = generate_branded_authorship_certificate_pdf(
        manuscript_title=job.title,
        author_name=job.author_name,
        publisher_name=branding.publisher_name or job.publisher_name,
        verdict=job.overall_verdict,
        human_authenticity_score=job.overall_human_score,
        ai_probability=job.overall_ai_probability,
        model_attribution=job.top_model_provenance,
        resonance_gap=job.resonance_gap,
        word_count=job.total_words,
        signature_hex=job.openssh_signature or "sha256_mock_sig_12345",
        chapter_breakdowns=chapter_dicts,
        submission_id=f"MS-{job.job_id[-6:].upper()}",
        brand_color_hex=branding.brand_color_hex,
        accent_color_hex=branding.accent_color_hex,
        signatory_name=branding.default_signatory_name,
        signatory_title=branding.default_signatory_title,
    )

    clean_filename = f"Certificate_{job.title[:25].replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{clean_filename}"'}
    )


# =========================================================================
# Commercial Stripe Billing & Multi-Tenant Endpoints
# =========================================================================
@app.post("/api/v1/billing/checkout")
async def create_stripe_checkout(req: CreateCheckoutRequest) -> Dict[str, Any]:
    """Generate Stripe Checkout Session for subscription upgrade."""
    stripe_mgr = get_stripe_manager()
    try:
        tier_enum = PlanTier(req.target_tier)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid plan tier '{req.target_tier}'. Choose 'author_pro' or 'publisher_b2b'.")

    try:
        checkout_res = stripe_mgr.create_checkout_session(
            tenant_id=req.tenant_id,
            target_tier=tier_enum,
            success_url=req.success_url or "https://manuscriptproof.com/billing/success",
            cancel_url=req.cancel_url or "https://manuscriptproof.com/billing/cancel",
        )
        return checkout_res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe Checkout creation failed: {str(e)}")


@app.post("/api/v1/billing/webhook")
async def stripe_webhook_handler(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
) -> Dict[str, Any]:
    """Inbound Stripe Webhook listener for subscription events."""
    stripe_mgr = get_stripe_manager()
    body_bytes = await request.body()

    if not stripe_mgr.verify_webhook_signature(body_bytes, stripe_signature or ""):
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook signature.")

    try:
        event_json = await request.json()
        result = stripe_mgr.handle_webhook_event(event_json)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook processing error: {str(e)}")


@app.post("/api/v1/billing/api-keys")
async def create_publisher_api_key(req: CreateApiKeyRequest) -> Dict[str, Any]:
    """Generate an API key for a Publisher B2B tenant."""
    stripe_mgr = get_stripe_manager()
    try:
        key_res = stripe_mgr.generate_publisher_api_key(req.tenant_id, req.key_name)
        return key_res
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/billing/tenant/{tenant_id}")
async def get_tenant_billing_profile(tenant_id: str) -> Dict[str, Any]:
    """Fetch tenant account info, current tier, and remaining word quotas."""
    stripe_mgr = get_stripe_manager()
    tenant = stripe_mgr.get_tenant_by_id(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found.")

    return {
        "tenant_id": tenant.tenant_id,
        "organization_name": tenant.organization_name,
        "owner_email": tenant.owner_email,
        "plan_tier": tenant.plan_tier.value,
        "tier_name": tenant.tier_config.name,
        "subscription_status": tenant.subscription_status.value,
        "words_processed_this_period": tenant.words_processed_this_period,
        "monthly_word_limit": tenant.tier_config.monthly_word_limit,
        "remaining_words": tenant.remaining_words(),
        "slushpile_api_access": tenant.tier_config.slushpile_api_access,
        "custom_branding_allowed": tenant.tier_config.custom_branding_allowed,
        "created_at": tenant.created_at,
    }


@app.post("/api/v1/billing/branding")
async def update_tenant_branding(req: UpdateBrandingRequest) -> Dict[str, Any]:
    """Update custom branding configuration for Publisher B2B accounts."""
    stripe_mgr = get_stripe_manager()
    tenant = stripe_mgr.get_tenant_by_id(req.tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant {req.tenant_id} not found.")

    if not tenant.tier_config.custom_branding_allowed:
        raise HTTPException(status_code=403, detail="Custom branding is only available on Author Pro and Publisher B2B tiers.")

    new_branding = CustomBrandingConfig(
        publisher_name=req.publisher_name,
        brand_color_hex=req.brand_color_hex,
        accent_color_hex=req.accent_color_hex,
        default_signatory_name=req.default_signatory_name,
        default_signatory_title=req.default_signatory_title,
    )
    stripe_mgr.update_custom_branding(req.tenant_id, new_branding)
    return {"status": "success", "branding": new_branding.__dict__}


# =========================================================================
# Core Forensics & Detection Endpoints
# =========================================================================
@app.post("/api/detect")
async def detect_ai_text(req: DetectRequest) -> Dict[str, Any]:
    """Run randomized Cloze Congruence AI detection on input text."""
    try:
        nim_client = NvidiaNIMClient(
            api_key=req.raw_api_key,
            encrypted_token=req.encrypted_token,
            fernet_key=req.fernet_key,
        )
        detector = ClozeCongruenceDetector(nim_client=nim_client)
        results = detector.analyze(
            text=req.text,
            mask_rate=req.mask_rate,
            num_passes=req.num_passes,
            model_name=req.model_name,
            temperature=req.temperature,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/api/detect/4pass")
async def detect_4pass(req: Detect4PassRequest) -> Dict[str, Any]:
    """Run full 4-Pass Multi-Scale Cloze Infilling with Pass-Adaptive Dynamic Gating."""
    try:
        client = OpenRouterClient(api_key=req.api_key, default_model=req.model_name)
        detector = ClozeCongruenceDetector(nim_client=client)
        
        result = detector.analyze_multi_trial_ensemble(
            text=req.text,
            pass1_trials=req.pass1_trials,
            pass2_trials=req.pass2_trials,
            pass3_trials=req.pass3_trials,
            pass4_trials=req.pass4_trials,
            model_name=req.model_name,
            temperature=req.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"4-Pass Detection failed: {str(e)}")


@app.post("/api/traceback")
async def dna_traceback(req: DNAAttributionRequest) -> Dict[str, Any]:
    """Run LLM DNA Fingerprinting and Foundation Model Attribution."""
    try:
        client = OpenRouterClient(api_key=req.api_key)
        engine = DNAAttributionEngine(client=client)
        
        result = engine.analyze(
            text=req.text,
            candidate_models=req.candidate_models
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DNA Attribution failed: {str(e)}")


@app.post("/api/humanize")
async def humanize_text(req: HumanizeRequest) -> Dict[str, Any]:
    """Rewrite text with high burstiness and anti-detection styling."""
    try:
        nim_client = NvidiaNIMClient(
            api_key=req.raw_api_key,
            encrypted_token=req.encrypted_token,
            fernet_key=req.fernet_key,
        )
        humanizer = TextHumanizer(nim_client=nim_client)
        result = humanizer.humanize(
            text=req.text,
            domain=req.domain,
            model_name=req.model_name,
            temperature=req.temperature,
        )
        
        detector = ClozeCongruenceDetector(nim_client=nim_client)
        humanized_detection = detector.analyze(result["humanized_text"]) if result["humanized_text"] else {}
        result["humanized_detection"] = humanized_detection

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Humanization failed: {str(e)}")


@app.post("/api/prompt")
async def generate_prompt(req: PromptGenerateRequest) -> Dict[str, Any]:
    """Generate anti-detection humanizer prompt template."""
    humanizer = TextHumanizer()
    bundle = humanizer.generate_humanize_prompt(
        domain=req.domain,
        target_audience=req.target_audience,
        additional_notes=req.additional_notes,
    )
    return bundle


@app.post("/api/encrypt")
async def encrypt_credentials_endpoint(req: EncryptRequest) -> Dict[str, Any]:
    """Encrypt an API key using Fernet key encryption."""
    try:
        fernet_key = req.fernet_key or generate_fernet_key()
        encrypted_token = encrypt_api_key(req.api_key, fernet_key)
        return {
            "status": "success",
            "fernet_secret_key": fernet_key,
            "encrypted_token": encrypted_token,
            "masked_key": mask_api_key(req.api_key),
            "instructions": {
                "dotenv": f"FERNET_SECRET_KEY={fernet_key}\nFERNET_ENCRYPTED_NVIDIA_API_KEY={encrypted_token}",
                "github_actions": "Add FERNET_SECRET_KEY and FERNET_ENCRYPTED_NVIDIA_API_KEY to Repository Secrets",
            },
        }
    except EncryptionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")


# Serve static frontend
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ManuscriptProof 1.0 Production API is active."}
