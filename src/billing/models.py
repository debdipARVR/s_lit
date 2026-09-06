"""Billing & Subscription Models for ManuscriptProof 1.0.

Defines commercial tier structures, quota constraints, and tenant profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class PlanTier(str, Enum):
    FREE_TRIAL = "free_trial"
    AUTHOR_PRO = "author_pro"
    PUBLISHER_B2B = "publisher_b2b"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"


@dataclass
class TierConfig:
    tier_id: PlanTier
    name: str
    price_usd_monthly: float
    stripe_price_id: str
    monthly_word_limit: int  # -1 represents unlimited
    max_concurrent_batch_jobs: int
    slushpile_api_access: bool
    custom_branding_allowed: bool
    multi_seat_allowed: bool
    support_level: str
    description: str


TIER_CATALOG: Dict[PlanTier, TierConfig] = {
    PlanTier.FREE_TRIAL: TierConfig(
        tier_id=PlanTier.FREE_TRIAL,
        name="Free Author Trial",
        price_usd_monthly=0.00,
        stripe_price_id="price_free_trial",
        monthly_word_limit=5_000,
        max_concurrent_batch_jobs=1,
        slushpile_api_access=False,
        custom_branding_allowed=False,
        multi_seat_allowed=False,
        support_level="Community",
        description="Ideal for single short-story or first chapter evaluation."
    ),
    PlanTier.AUTHOR_PRO: TierConfig(
        tier_id=PlanTier.AUTHOR_PRO,
        name="Author Pro ($19.99/mo)",
        price_usd_monthly=19.99,
        stripe_price_id="price_author_pro_monthly",
        monthly_word_limit=150_000,
        max_concurrent_batch_jobs=5,
        slushpile_api_access=False,
        custom_branding_allowed=True,
        multi_seat_allowed=False,
        support_level="Priority Email (24h)",
        description="Full novel scanning for Indie & Amazon KDP Authors. PDF Certificates & Chapter Heatmaps."
    ),
    PlanTier.PUBLISHER_B2B: TierConfig(
        tier_id=PlanTier.PUBLISHER_B2B,
        name="Publisher B2B Enterprise ($499/mo)",
        price_usd_monthly=499.00,
        stripe_price_id="price_publisher_b2b_monthly",
        monthly_word_limit=-1,  # Unlimited
        max_concurrent_batch_jobs=50,
        slushpile_api_access=True,
        custom_branding_allowed=True,
        multi_seat_allowed=True,
        support_level="Dedicated Account Manager & SLA",
        description="For Big-5 Publishing Houses & Literary Agencies. Automated Slush-Pile Webhook API & Custom Co-Branded Certificates."
    ),
}


@dataclass
class CustomBrandingConfig:
    publisher_name: str = "ManuscriptProof Enterprise"
    brand_color_hex: str = "#1e1b4b"
    accent_color_hex: str = "#6366f1"
    logo_base64: Optional[str] = None
    default_signatory_name: str = "Senior Acquisitions Editor"
    default_signatory_title: str = "Manuscript Review Board"
    custom_disclaimer: Optional[str] = None


@dataclass
class TenantAccount:
    tenant_id: str
    organization_name: str
    owner_email: str
    plan_tier: PlanTier = PlanTier.FREE_TRIAL
    subscription_status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    current_period_start: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    current_period_end: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30 * 86400)))
    words_processed_this_period: int = 0
    total_evaluations: int = 0
    branding: CustomBrandingConfig = field(default_factory=CustomBrandingConfig)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def tier_config(self) -> TierConfig:
        return TIER_CATALOG.get(self.plan_tier, TIER_CATALOG[PlanTier.FREE_TRIAL])

    def can_process_words(self, words: int) -> bool:
        """Check if tenant has remaining word quota for this period."""
        if self.tier_config.monthly_word_limit == -1:
            return True
        return (self.words_processed_this_period + words) <= self.tier_config.monthly_word_limit

    def remaining_words(self) -> int:
        """Remaining words quota before renewal."""
        if self.tier_config.monthly_word_limit == -1:
            return 999_999_999
        return max(0, self.tier_config.monthly_word_limit - self.words_processed_this_period)
