"""
Core logic for the resume screener: PDF text extraction, candidate info
extraction, skill matching, and a small logistic regression model that
scores a candidate's overall fit.

This is a cleaned-up, reusable version of the exploratory code from
Main.py / Main.ipynb: no hardcoded file paths, no mysql credentials,
and everything works on in-memory uploaded files instead of local disk paths.
"""

import io
import random
import re
from typing import List, Tuple

import numpy as np
import pdfplumber
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Default skills vocabulary (can be overridden per-request)
# ---------------------------------------------------------------------------
DEFAULT_SKILLS = [
    "python", "django", "flask", "fastapi", "sql", "mysql", "postgresql",
    "machine learning", "deep learning", "nlp", "computer vision",
    "javascript", "typescript", "react", "node", "java", "c++", "c#",
    "html", "css", "git", "docker", "kubernetes", "aws", "azure", "gcp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
]

EDUCATION_KEYWORDS_ADVANCED = ["mca", "m.tech", "mtech", "master", "phd", "m.sc", "msc"]
EDUCATION_KEYWORDS_BASIC = ["bca", "b.tech", "btech", "bachelor", "b.sc", "bsc", "degree"]

EXPERIENCE_KEYWORDS = ["intern", "internship", "experience", "trainee", "worked at", "employed"]
PROJECT_KEYWORDS = ["project", "built", "developed", "implemented", "designed"]

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+\s*@\s*[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}|\+?\d{10,13}")
YEARS_RE = re.compile(r"(\d+)\+?\s*(?:years|yrs)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# PDF text extraction (works on in-memory bytes, not a Windows file path)
# ---------------------------------------------------------------------------
def extract_text_from_bytes(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ---------------------------------------------------------------------------
# Candidate info extraction (best-effort heuristics)
# ---------------------------------------------------------------------------
def extract_candidate_info(text: str) -> dict:
    email_match = EMAIL_RE.search(text)
    phone_match = PHONE_RE.search(text)

    name = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that look like contact info / headers
        if EMAIL_RE.search(line) or PHONE_RE.search(line):
            continue
        if len(line.split()) <= 5 and line.replace(" ", "").isalpha():
            name = line
            break

    email = email_match.group(0).replace(" ", "") if email_match else None
    phone = phone_match.group(0) if phone_match else None

    return {"name": name, "email": email, "phone": phone}


# ---------------------------------------------------------------------------
# Skill matching between a resume and a job description
# ---------------------------------------------------------------------------
def calculate_skill_match(resume_text: str, jd_text: str, skills: List[str]) -> dict:
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    required_skills = [s for s in skills if s.lower() in jd_lower]
    matched_skills = [s for s in required_skills if s.lower() in resume_lower]
    missing_skills = [s for s in required_skills if s not in matched_skills]

    score = (len(matched_skills) / len(required_skills) * 100) if required_skills else 0.0

    return {
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "match_score": round(score, 2),
    }


def _keyword_hits(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    return sum(1 for k in keywords if k in text_lower)


def compute_score_breakdown(resume_text: str, skill_score: float) -> dict:
    """Turn a resume + skill match score into 4 features (0-100 each),
    similar in spirit to the [skills, education, experience, projects]
    feature vector used in the original notebook."""

    edu_hits_adv = _keyword_hits(resume_text, EDUCATION_KEYWORDS_ADVANCED)
    edu_hits_basic = _keyword_hits(resume_text, EDUCATION_KEYWORDS_BASIC)
    if edu_hits_adv:
        education_score = 100.0
    elif edu_hits_basic:
        education_score = 65.0
    else:
        education_score = 30.0

    years_match = YEARS_RE.search(resume_text)
    exp_hits = _keyword_hits(resume_text, EXPERIENCE_KEYWORDS)
    if years_match:
        years = int(years_match.group(1))
        experience_score = min(100.0, 40.0 + years * 15.0)
    elif exp_hits >= 2:
        experience_score = 65.0
    elif exp_hits == 1:
        experience_score = 45.0
    else:
        experience_score = 20.0

    proj_hits = _keyword_hits(resume_text, PROJECT_KEYWORDS)
    project_score = min(100.0, proj_hits * 20.0)

    return {
        "skill_score": round(skill_score, 2),
        "education_score": round(education_score, 2),
        "experience_score": round(experience_score, 2),
        "project_score": round(project_score, 2),
    }


# ---------------------------------------------------------------------------
# ML model: logistic regression over the 4 score features.
# The original notebook trained on 3 hand-typed rows. Here we generate a
# larger synthetic training set with a clear (but noisy) decision rule so
# the model behaves sensibly across the whole range of inputs, and we train
# it once at import time.
# ---------------------------------------------------------------------------
_MODEL: LogisticRegression = None


def _generate_synthetic_training_data(n: int = 400, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        skill = rng.uniform(0, 100)
        edu = rng.uniform(0, 100)
        exp = rng.uniform(0, 100)
        proj = rng.uniform(0, 100)

        weighted = 0.45 * skill + 0.15 * edu + 0.25 * exp + 0.15 * proj
        noisy = weighted + rng.gauss(0, 8)
        label = 1 if noisy >= 65 else 0

        X.append([skill, edu, exp, proj])
        y.append(label)
    return np.array(X), np.array(y)


def get_model() -> LogisticRegression:
    global _MODEL
    if _MODEL is None:
        X, y = _generate_synthetic_training_data()
        model = LogisticRegression()
        model.fit(X, y)
        _MODEL = model
    return _MODEL


def predict_fit(breakdown: dict) -> Tuple[float, str]:
    model = get_model()
    features = [[
        breakdown["skill_score"],
        breakdown["education_score"],
        breakdown["experience_score"],
        breakdown["project_score"],
    ]]
    probability = float(model.predict_proba(features)[0][1])
    verdict = "Recommended" if probability >= 0.5 else "Not Recommended"
    return round(probability, 4), verdict
