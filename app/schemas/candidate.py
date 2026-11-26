from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.position import FacetedVectors

ExtractionStatus = Literal["completed", "needs_ocr", "failed"]
ExtractionMethod = Literal["llm", "fallback_heuristic", "none"]


class CandidateMetadata(BaseModel):
    """Sanitized DB fields submitted alongside the CV file."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1)
    city: str | None = None
    gpa: float | None = Field(default=None, ge=0.0, le=100.0)
    year_of_study: int | None = Field(default=None, ge=1)
    full_time_commitment: bool = True
    program: str = ""
    field: str = ""
    major: str = ""
    technical_skills: list[str] = Field(default_factory=list)
    additional_info: str = ""


class ParsedCV(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[str] = Field(default_factory=list)
    experience_summary: str = ""
    education_summary: str = ""
    projects_summary: str = ""
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    cv_completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CandidateIngestResponse(BaseModel):
    candidate_id: str
    cv_hash: str = Field(..., min_length=64, max_length=64)
    extraction_status: ExtractionStatus
    extraction_method: ExtractionMethod
    extraction_error: str | None = None
    parsed_cv: ParsedCV | None = None
    vectors: FacetedVectors | None = None
    model_version: str


class CandidateForAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1)
    metadata: CandidateMetadata
    parsed_cv: ParsedCV
    vectors: FacetedVectors

    @model_validator(mode="after")
    def validate_matching_ids(self) -> "CandidateForAllocation":
        if self.candidate_id != self.metadata.candidate_id:
            raise ValueError("candidate_id must match metadata.candidate_id")
        return self