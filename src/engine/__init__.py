"""Engine package for Cloze Congruence AI Text Detection, LLM DNA Attribution, and Humanization."""

from .cloze_masker import ClozeMasker, ClozeMaskResult, MaskedSpan
from .detector import ClozeCongruenceDetector
from .dna_attribution import (
    DNAAttributionEngine,
    extract_dna_features,
    compute_dna_profile_similarity,
    MODEL_CLASSES,
    MODEL_DISPLAY_NAMES,
    BASELINE_DNA_PROFILES,
)
from .humanizer import TextHumanizer, HUMANIZER_MODES
from .metrics import (
    calculate_burstiness,
    classify_span_congruence,
    compute_ai_probability,
    compute_lexical_similarity,
    compute_semantic_congruence,
)
from .nim_client import NvidiaNIMClient, NVIDIA_MODELS
from .openrouter_client import (
    OpenRouterClient,
    OPENROUTER_MODELS,
    DEFAULT_OPENROUTER_MODEL,
)

__all__ = [
    "ClozeMasker",
    "ClozeMaskResult",
    "MaskedSpan",
    "ClozeCongruenceDetector",
    "DNAAttributionEngine",
    "extract_dna_features",
    "compute_dna_profile_similarity",
    "MODEL_CLASSES",
    "MODEL_DISPLAY_NAMES",
    "BASELINE_DNA_PROFILES",
    "TextHumanizer",
    "HUMANIZER_MODES",
    "NvidiaNIMClient",
    "NVIDIA_MODELS",
    "OpenRouterClient",
    "OPENROUTER_MODELS",
    "DEFAULT_OPENROUTER_MODEL",
    "compute_lexical_similarity",
    "compute_semantic_congruence",
    "calculate_burstiness",
    "compute_ai_probability",
    "classify_span_congruence",
]
