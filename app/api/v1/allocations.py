from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import verify_api_key
from app.schemas.allocation import (
    AllocationExplainRequest,
    AllocationExplainResponse,
    AllocationSolveRequest,
    AllocationSolveResponse,
)

router = APIRouter(
    prefix="/api/v1/allocations",
    tags=["Allocations"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/solve", response_model=AllocationSolveResponse)
async def solve_allocations(payload: AllocationSolveRequest) -> AllocationSolveResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Eligibility gate, scoring, and CP-SAT solver will be implemented in Phases 5-6.",
    )


@router.post("/explain", response_model=AllocationExplainResponse)
async def explain_allocations(payload: AllocationExplainRequest) -> AllocationExplainResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Post-allocation explainer will be implemented in Phase 6.",
    )