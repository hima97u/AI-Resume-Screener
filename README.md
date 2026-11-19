
# AI Resume Screener

A Python-based desktop application that screens and ranks PDF resumes against a job description using NLP and machine learning techniques.

## Features

- Upload multiple PDF resumes or an entire folder.
- Match candidates using TF-IDF and cosine similarity.
- Analyze job descriptions using NLTK.
- Automatically detect candidate names, experience, and education.
- Rank candidates based on an overall matching score.
- Identify matched and missing skills.
- Search candidates and view detailed evaluation results.

## Tech Stack

- **Language:** Python 3.14+
- **GUI:** Tkinter
- **NLP:** NLTK
- **ML:** Scikit-learn (TF-IDF, Cosine Similarity)
- **PDF Processing:** PyPDF

## Installation

```bash
git clone https://github.com/hima97u/AI-Resume-Screener.git
cd AI-Resume-Screener

python -m venv .venv
```

Activate the virtual environment:

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
python main.py
```

## How to Use

1. Enter the job description.
2. Add the required skills (comma-separated).
3. Upload candidate PDF resumes or select a folder.
4. Click **Screen & Rank Candidates**.
5. View rankings, matching scores, and skill analysis.

## Note

The application supports text-based PDFs. Scanned/image-only resumes are not supported because OCR is not implemented. Candidate scores are similarity-based estimates and should be used as decision-support metrics.
