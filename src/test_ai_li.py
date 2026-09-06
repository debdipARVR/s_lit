"""Evaluation metrics and congruency criteria for AI text detection."""

try:
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval, BERTScoreMetric
except ImportError:
    pass

from .deepeval_model_connect import NvidiaLLM_Understanding


def create_ai_detection_metrics(model_name: str = "zhipuai/glm-4-plus"):
    """Create GEval congruency metrics for AI detection comparison."""
    evaluator = NvidiaLLM_Understanding(model_name=model_name)
    
    congruency_metric = GEval(
        name="Cloze_Congruency",
        criteria="""
        Evaluate how congruent (aligned) these two texts are across four dimensions:
        1. Thematic Alignment: Do both texts cover the same core themes and subject matter?
        2. Structural Parallelism: Do the texts follow similar organizational logic and syntax?
        3. Factual Consistency: Are facts and statements compatible (not contradictory)?
        4. Semantic Similarity: Do corresponding points convey equivalent meaning?
        
        Provide a score from 0-1 and explain which dimensions show alignment vs divergence.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
        model=evaluator,
    )
    return congruency_metric