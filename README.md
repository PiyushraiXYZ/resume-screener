# Case File — Resume Screener

A FastAPI backend + static HTML frontend, rebuilt from your `Main.py` / `Main.ipynb`
prototype. It keeps your original approach (PDF text extraction → skill keyword
matching → logistic regression scoring) but turns it into a real, callable API with:

- No more hardcoded `C:\Users\Admin\Downloads\...` paths — everything works on
  uploaded files.
- No MySQL credentials hardcoded in the code.
- A trained logistic regression model that scores 4 real signals (skill match,
  education, experience, projects) instead of 3 hand-typed training rows.
- Endpoints for both a single resume-vs-JD analysis and ranking many resumes
  against one job description.
- A "case file" themed frontend to use it from a browser.

## Project structure

```
resume-screener/
├── backend/
│   ├── main.py            FastAPI app (routes)
│   ├── resume_utils.py    PDF parsing, skill matching, ML model
│   ├── schema.py          Pydantic response models
│   └── requirements.txt
└── frontend/
    └── index.html         Single-file frontend (no build step)
```

## Running the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

### Endpoints

- `GET /api/health` — health check
- `POST /api/analyze` — score one resume against one job description
  - form fields: `resume` (PDF, required), `jd_file` (PDF) **or** `jd_text` (string),
    `skills` (optional comma-separated custom skill list)
- `POST /api/rank` — score multiple resumes against one job description, sorted
  best-fit first
  - form fields: `resumes` (multiple PDFs, required), `jd_file`/`jd_text`, `skills`

## Running the frontend

It's a single static HTML file — no build step needed.

```bash
cd frontend
python3 -m http.server 8080
```

Then open http://localhost:8080 in a browser. There's an "API base" field at the
top of the page (defaults to `http://localhost:8000`) in case your backend runs
somewhere else. The CORS policy on the backend is currently wide open (`*`) to
make local development easy — tighten `allow_origins` in `main.py` before
deploying this anywhere public.

## How the scoring works

1. Text is extracted from both PDFs with `pdfplumber`.
2. `calculate_skill_match` checks which skills from the list actually appear in
   the JD (`required_skills`), and which of those also appear in the resume
   (`matched_skills`) → a 0–100 match score.
3. `compute_score_breakdown` derives 3 more heuristic scores from the resume
   text: education level, experience signals (years mentioned, internship/experience
   keywords), and project mentions.
4. Those 4 scores feed a `LogisticRegression` model (trained once at startup on
   400 synthetic, weighted-and-noisy examples) that outputs a fit probability
   and a Recommended / Not Recommended verdict.

This is still a heuristic/demo-grade system, not a production hiring tool —
keyword matching won't catch synonyms or context, and the "training data" is
synthetic. If you want to grow this further, the natural next steps are:
a labeled dataset of real hiring outcomes, smarter skill matching (e.g. embeddings
instead of exact keyword hits), and a persistent database instead of stateless
requests.
