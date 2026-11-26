from pydantic import BaseModel, ConfigDict, Field, model_validator


class FacetedVectors(BaseModel):
    """Three L2-normalized embedding vectors aligned across candidates and positions."""

    skills: list[float] = Field(..., min_length=1)
    academic: list[float] = Field(..., min_length=1)
    practical: list[float] = Field(..., min_length=1)


class PositionConstraints(BaseModel):
    """Deterministic hard-eligibility rules evaluated before scoring."""

    allowed_cities: list[str] = Field(default_factory=list)
    min_gpa: float | None = Field(default=None, ge=0.0, le=100.0)
    min_year_of_study: int | None = Field(default=None, ge=1)
    requires_full_time_commitment: bool = False
    required_majors: list[str] = Field(default_factory=list)
    must_have_keywords: list[str] = Field(default_factory=list)


class PositionEmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    department: str | None = None
    description: str = ""
    responsibilities: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    target_majors: list[str] = Field(default_factory=list)
    academic_level: str = ""
    experience_level: str = ""


class PositionEmbedResponse(BaseModel):
    position_id: str
    vectors: FacetedVectors
    model_version: str


class PositionForAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: str = ""
    min_seats: int = Field(default=0, ge=0)
    max_seats: int = Field(..., ge=1)
    constraints: PositionConstraints = Field(default_factory=PositionConstraints)
    vectors: FacetedVectors

    @model_validator(mode="after")
    def validate_seat_bounds(self) -> "PositionForAllocation":
        if self.min_seats > self.max_seats:
            raise ValueError("min_seats cannot exceed max_seats")
        return self