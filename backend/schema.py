"""
Pydantic models shared across the resume screener API.
(This fills in what was previously an empty resume_schema.py)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class SkillMatch(BaseModel):
    required_skills: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    match_score: float = 0.0  # 0-100


class ScoreBreakdown(BaseModel):
    skill_score: float
    education_score: float
    experience_score: float
    project_score: float


class AnalyzeResponse(BaseModel):
    candidate: CandidateInfo
    skills: SkillMatch
    breakdown: ScoreBreakdown
    ml_probability: float  # 0-1, model's estimated probability of a good fit
    ml_verdict: str        # "Recommended" / "Not Recommended"
    resume_preview: str


class RankedCandidate(BaseModel):
    filename: str
    candidate: CandidateInfo
    skills: SkillMatch
    breakdown: ScoreBreakdown
    ml_probability: float
    ml_verdict: str


class RankResponse(BaseModel):
    jd_required_skills: List[str]
    ranked: List[RankedCandidate]
