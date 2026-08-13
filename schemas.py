"""Pydantic schemas: the structured-output contract the LLM must satisfy.

Main points:
This is how I enforce that an LLM's free-text output becomes a validated, 
machine-readable object. If the model returns something malformed, the
constructor raises the ValidationError and the agent loop re-prompts (see agent.py).
Same technique I used in Thematica between the coding/theming/insight stages.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class AnalysisStep(BaseModel):
    """One planned analysis: a specific question plus the pandas code to answer it."""

    question: str = Field(..., description="The specific analytical question being asked")
    pandas_code: str = Field(..., description="pandas code that assigns its answer to `result`")

    @field_validator("pandas_code")
    @classmethod
    def must_assign_result(cls, v: str) -> str:
        # We require a `result` variable so the sandbox knows what to return.
        # This validator is a cheap guardrail that catches a whole class of bad output
        # before we ever execute anything.
        if "result" not in v:
            raise ValueError("pandas_code must assign to a variable named `result`")
        return v


class Insight(BaseModel):
    """A grounded finding, produced only after code has actually executed."""

    finding: str = Field(..., description="Plain-language insight, 1-2 sentences")
    supporting_question: str
    supporting_code: str
    result_summary: str = Field(..., description="The concrete numbers backing the finding")
    confidence: Literal["low", "medium", "high"]

    novelty: Literal["expected", "surprising"] = Field(
        ..., description="Whether this is a known pattern or a potentially unseen one"
    )
