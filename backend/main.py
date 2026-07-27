"""
Resume Screener API
--------------------
FastAPI service built from the Main.py / Main.ipynb prototype.

Endpoints:
  GET  /api/health           - health check
  POST /api/analyze          - score ONE resume against ONE job description
  POST /api/rank             - score MULTIPLE resumes against ONE job description,
                                returned ranked best-to-worst

Run locally:
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

Then open frontend/index.html in a browser (it calls http://localhost:8000).
"""

from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from resume_utils import (
    DEFAULT_SKILLS,
    calculate_skill_match,
    compute_score_breakdown,
    extract_candidate_info,
    extract_text_from_bytes,
    predict_fit,
)
from schema import (
    AnalyzeResponse,
    CandidateInfo,
    RankedCandidate,
    RankResponse,
    ScoreBreakdown,
    SkillMatch,
)

app = FastAPI(
    title="Resume Screener API",
    description="Extracts resume text, matches skills against a job description, "
                "and scores candidate fit with a logistic regression model.",
    version="1.0.0",
)

# Allow the static frontend (opened via file:// or any localhost port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_skills(skills_csv: Optional[str]) -> List[str]:
    if not skills_csv:
        return DEFAULT_SKILLS
    parsed = [s.strip() for s in skills_csv.split(",") if s.strip()]
    return parsed or DEFAULT_SKILLS


async def _read_pdf_upload(upload: UploadFile) -> str:
    if not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"{upload.filename} must be a PDF file")
    file_bytes = await upload.read()
    try:
        text = extract_text_from_bytes(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {upload.filename}: {exc}")
    if not text.strip():
        raise HTTPException(status_code=422, detail=f"No extractable text found in {upload.filename}")
    return text


def _score_resume(resume_text: str, jd_text: str, skills: List[str]) -> dict:
    match = calculate_skill_match(resume_text, jd_text, skills)
    breakdown = compute_score_breakdown(resume_text, match["match_score"])
    probability, verdict = predict_fit(breakdown)
    candidate = extract_candidate_info(resume_text)
    return {
        "candidate": candidate,
        "skills": match,
        "breakdown": breakdown,
        "ml_probability": probability,
        "ml_verdict": verdict,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(..., description="Candidate resume, PDF"),
    jd_file: Optional[UploadFile] = File(None, description="Job description, PDF"),
    jd_text: Optional[str] = Form(None, description="Job description as plain text"),
    skills: Optional[str] = Form(None, description="Comma-separated custom skill list"),
):
    if not jd_file and not jd_text:
        raise HTTPException(status_code=400, detail="Provide either jd_file or jd_text")

    resume_text = await _read_pdf_upload(resume)
    jd_full_text = await _read_pdf_upload(jd_file) if jd_file else jd_text

    result = _score_resume(resume_text, jd_full_text, _parse_skills(skills))

    return AnalyzeResponse(
        candidate=CandidateInfo(**result["candidate"]),
        skills=SkillMatch(**result["skills"]),
        breakdown=ScoreBreakdown(**result["breakdown"]),
        ml_probability=result["ml_probability"],
        ml_verdict=result["ml_verdict"],
        resume_preview=resume_text[:500],
    )


@app.post("/api/rank", response_model=RankResponse)
async def rank(
    resumes: List[UploadFile] = File(..., description="Multiple candidate resumes, PDF"),
    jd_file: Optional[UploadFile] = File(None, description="Job description, PDF"),
    jd_text: Optional[str] = Form(None, description="Job description as plain text"),
    skills: Optional[str] = Form(None, description="Comma-separated custom skill list"),
):
    if not jd_file and not jd_text:
        raise HTTPException(status_code=400, detail="Provide either jd_file or jd_text")

    jd_full_text = await _read_pdf_upload(jd_file) if jd_file else jd_text
    skill_list = _parse_skills(skills)

    ranked: List[RankedCandidate] = []
    for upload in resumes:
        try:
            resume_text = await _read_pdf_upload(upload)
        except HTTPException as exc:
            # Skip unreadable files but keep processing the rest of the batch.
            continue
        result = _score_resume(resume_text, jd_full_text, skill_list)
        ranked.append(
            RankedCandidate(
                filename=upload.filename,
                candidate=CandidateInfo(**result["candidate"]),
                skills=SkillMatch(**result["skills"]),
                breakdown=ScoreBreakdown(**result["breakdown"]),
                ml_probability=result["ml_probability"],
                ml_verdict=result["ml_verdict"],
            )
        )

    ranked.sort(key=lambda c: c.ml_probability, reverse=True)
    required_skills = [s for s in skill_list if s.lower() in jd_full_text.lower()]

    return RankResponse(jd_required_skills=required_skills, ranked=ranked)
