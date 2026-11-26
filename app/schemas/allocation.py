from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.candidate import CandidateForAllocation
from app.schemas.position import PositionForAllocation

UnassignedReason = Literal["all_ineligible", "below_threshold", "capacity_full"]


class LockedOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1)
    position_id: str = Field(..., min_length=1)


class ScoringWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills_semantic: float = Field(default=0.35, ge=0.0, le=1.0)
    academic_semantic: float = Field(default=0.15, ge=0.0, le=1.0)
    practical_semantic: float = Field(default=0.20, ge=0.0, le=1.0)
    required_skills_lexical: float = Field(default=0.20, ge=0.0, le=1.0)
    preferred_skills_lexical: float = Field(default=0.10, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "ScoringWeights":
        total = (
            self.skills_semantic
            + self.academic_semantic
            + self.practical_semantic
            + self.required_skills_lexical
            + self.preferred_skills_lexical
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0 (got {total:.4f})")
        return self


class AllocationSolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateForAllocation] = Field(..., min_length=1)
    positions: list[PositionForAllocation] = Field(..., min_length=1)
    locked_overrides: list[LockedOverride] = Field(default_factory=list)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    min_match_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    top_k_per_candidate: int = Field(default=12, ge=1, le=50)


class ScoreBreakdown(BaseModel):
    skills_semantic: float
    academic_semantic: float
    practical_semantic: float
    required_skills_lexical: float
    preferred_skills_lexical: float


class AlternativePlacement(BaseModel):
    position_id: str
    score: float
    score_breakdown: ScoreBreakdown


class AssignmentResult(BaseModel):
    candidate_id: str
    position_id: str
    score: float
    score_breakdown: ScoreBreakdown
    alternatives: list[AlternativePlacement] = Field(default_factory=list, max_length=2)
    is_locked: bool = False
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class UnfilledPosition(BaseModel):
    position_id: str
    min_seats: int
    max_seats: int
    assigned_count: int
    deficit: int


class UnassignedCandidate(BaseModel):
    candidate_id: str
    reason: UnassignedReason
    alternatives: list[AlternativePlacement] = Field(default_factory=list, max_length=2)
    disqualification_summary: dict[str, list[str]] = Field(default_factory=dict)


class AllocationSolveResponse(BaseModel):
    solver_status: Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE"]
    assignments: list[AssignmentResult]
    unfilled_positions: list[UnfilledPosition]
    unassigned_candidates: list[UnassignedCandidate]
    model_version: str


class ExplanationItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: CandidateForAllocation
    assigned_position: PositionForAllocation | None = None
    score: float | None = None
    score_breakdown: ScoreBreakdown | None = None
    alternatives: list[PositionForAllocation] = Field(default_factory=list, max_length=2)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class AllocationExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ExplanationItemInput] = Field(..., min_length=1)


class CandidateExplanation(BaseModel):
    candidate_id: str
    position_id: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    uncertainty: str | None = None


class AllocationExplainResponse(BaseModel):
    explanations: list[CandidateExplanation]