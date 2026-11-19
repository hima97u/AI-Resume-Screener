import sys

if sys.version_info < (3, 14):
    raise SystemExit(f"Python 3.14+ is required (found {sys.version.split()[0]}).")

import os
import re
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. NLTK BOOTSTRAP & PDF EXTRACTION
# =============================================================================

NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
    ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
    ("corpora/words", "words"),
]


def ensure_nltk_resources() -> None:
    """Download any missing NLTK datasets (compatible with NLTK 3.8 and 3.9+)."""
    for path, pkg in NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


# Download before class-level definitions access WordNet constants.
ensure_nltk_resources()


def extract_pdf_text(filepath: str) -> str:
    """Extract and normalize text from a PDF resume."""
    reader = PdfReader(filepath)
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise ValueError(f"Encrypted PDF cannot be opened: {exc}") from exc
        if decrypted == 0:
            raise ValueError("PDF is password-protected and cannot be read.")

    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    raw = raw.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(
        cleaned for line in raw.split("\n")
        if (cleaned := re.sub(r"[ \t]+", " ", line).strip())
    )


# =============================================================================
# 2. DATA MODEL
# =============================================================================

@dataclass
class CandidateResult:
    filepath: str
    filename: str
    candidate_name: str
    overall_score: float = 0.0
    skill_score: float = 0.0
    jd_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    detected_experience_years: float | None = None
    detected_education: str = "Not Detected"
    warning: str = ""


# =============================================================================
# 3. NLP & TF-IDF SCREENING ENGINE
# =============================================================================

class ResumeScreenerEngine:
    """Screens resumes via TF-IDF skill matching, NLTK JD similarity, experience, and education."""

    IGNORED_NAME_WORDS = {
        # Common resume headers
        "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective",
        "contact", "personal", "details", "information", "experience", "education",
        "skills", "projects", "certifications", "achievements", "about", "me",
        "professional", "work", "history", "employment", "portfolio", "page",
        # Job title words
        "engineer", "developer", "manager", "analyst", "scientist", "architect",
        "consultant", "intern", "specialist", "director", "lead", "senior",
        "junior", "associate", "coordinator", "administrator", "designer",
        "executive", "officer", "president", "head", "vp", "full", "stack",
        "software", "data", "product", "project", "human", "resources", "hr",
        "marketing", "sales", "finance", "business", "operations", "cloud",
        "devops", "qa", "tester", "researcher", "phd", "mba",
    }

    EDUCATION_LEVELS: list[tuple[int, str, re.Pattern]] = [
        (4, "Doctorate / PhD", re.compile(r"\b(?:ph\.?d\.?|doctorate|doctoral|d\.phil)\b", re.I)),
        (3, "Master's Degree", re.compile(r"\b(?:master'?s?|m\.?s\.?c?|m\.?tech|m\.?e\.?|mba|m\.?b\.?a\.?|mca|m\.?a\.?|post\s*graduate)\b", re.I)),
        (2, "Bachelor's Degree", re.compile(r"\b(?:bachelor'?s?|b\.?s\.?c?|b\.?tech|b\.?e\.?|b\.?a\.?|bca|b\.?com|bba|undergraduate)\b", re.I)),
        (1, "Diploma / Associate", re.compile(r"\b(?:associate'?s?\s+degree|diploma|polytechnic)\b", re.I)),
    ]

    JD_YEAR_PATTERNS = [
        re.compile(r"(?:minimum|min\.?|at\s+least)\s+(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)", re.I),
        re.compile(r"(\d{1,2}(?:\.\d)?)\s*(?:\+|plus)?\s*(?:-\s*\d{1,2})?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp|relevant|hands-on|professional|industry)", re.I),
        re.compile(r"(?:experience|exp)\s*[:\-]?\s*(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:-\s*\d{1,2})?\s*(?:years?|yrs?)", re.I),
    ]

    EXP_EXPLICIT_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[\w\s]{0,20})?(?:experience|exp)\b", re.I)
    DATE_RANGE_RE = re.compile(r"\b(19[89]\d|20[0-2]\d)\s*(?:[-–—]|to)\s*(19[89]\d|20[0-2]\d|present|current|now|till\s*date)\b", re.I)
    EDU_LINE_RE = re.compile(r"\b(?:university|college|school|bachelor|master|b\.?tech|m\.?tech|gpa|cgpa)\b", re.I)
    EXP_START_RE = re.compile(r"^\s*(?:work\s+|professional\s+)?(?:experience|employment(?:\s+history)?|career\s+history)\s*:?\s*$", re.I)
    EXP_STOP_RE = re.compile(r"^\s*(?:education|academic|certifications?|skills?|publications?|awards?|languages?)\s*:?\s*$", re.I)
    FILENAME_NOISE_RE = re.compile(r"\b(?:resume|cv|curriculum|vitae|final|updated|latest|new|copy|profile|20\d{2})\b", re.I)

    POS_MAP = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}

    def __init__(self) -> None:
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = {
                "a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "are", "was", "were",
                "be", "been", "being", "this", "that", "it", "we", "you", "they",
            }

    # --- Text Preprocessing ---

    def preprocess_text_nltk(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            tokens = re.findall(r"[a-z0-9+#.]+", text.lower())

        clean = [t for t in tokens if len(t) > 1 and t not in self.stop_words and re.search(r"[a-z0-9]", t)]
        if not clean:
            return ""

        try:
            return " ".join(
                self.lemmatizer.lemmatize(tok, self.POS_MAP.get(tag[0], wordnet.NOUN))
                for tok, tag in pos_tag(clean)
            )
        except LookupError:
            return " ".join(clean)

    # --- Candidate Name Extraction ---

    def extract_candidate_name(self, resume_text: str, filename: str) -> str:
        header_lines = self._get_header_lines(resume_text)[:6]

        for extractor in (self._extract_name_ner, self._extract_name_heuristic):
            for line in header_lines:
                if name := extractor(line):
                    return self._format_name(name)

        return self._name_from_filename(filename)

    @staticmethod
    def _get_header_lines(text: str) -> list[str]:
        lines: list[str] = []
        for raw in text.splitlines()[:15]:
            line = re.sub(r"\s+", " ", raw).strip()
            lower = line.lower()
            if not (3 <= len(line) <= 65):
                continue
            if any(tok in lower for tok in ("@", "http", "www.", "linkedin", "github")):
                continue
            if len(re.findall(r"\d", line)) >= 4:
                continue
            if segment := re.split(r"[|•·–—]| - ", line)[0].strip():
                lines.append(segment)
        return lines

    def _extract_name_ner(self, line: str) -> str | None:
        probe = line.title() if line.isupper() else line
        try:
            tree = ne_chunk(pos_tag(word_tokenize(probe)), binary=False)
            for node in tree:
                if isinstance(node, Tree) and node.label() == "PERSON":
                    tokens = [tok for tok, _ in node.leaves()]
                    if self._is_valid_name_tokens(tokens):
                        return " ".join(tokens)
        except Exception:
            pass
        return None

    def _extract_name_heuristic(self, line: str) -> str | None:
        tokens = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", line).split()
        if not self._is_valid_name_tokens(tokens):
            return None
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z'\-\.]*", t) for t in tokens):
            return None

        probe = [t.title() if t.isupper() else t for t in tokens]
        try:
            nnp_count = sum(1 for _, tag in pos_tag(probe) if tag in ("NNP", "NNPS"))
            return " ".join(probe) if nnp_count >= max(1, len(tokens) - 1) else None
        except Exception:
            return " ".join(probe)

    def _is_valid_name_tokens(self, tokens: list[str]) -> bool:
        if not (2 <= len(tokens) <= 4):
            return False
        words = [re.sub(r"[^a-z]", "", t.lower()) for t in tokens]
        return all(words) and not any(w in self.IGNORED_NAME_WORDS for w in words)

    @classmethod
    def _name_from_filename(cls, filename: str) -> str:
        stem = re.sub(r"[_\-.]+", " ", Path(filename).stem)
        stem = cls.FILENAME_NOISE_RE.sub("", stem)
        stem = re.sub(r"\d+", "", stem)
        stem = re.sub(r"\s+", " ", stem).strip()
        return stem.title() if stem else Path(filename).stem

    @staticmethod
    def _format_name(name: str) -> str:
        return " ".join(
            p.upper() if len(p) <= 2 and p.endswith(".") else p.capitalize()
            for p in name.split()
        )

    # --- Experience & Education Extraction ---

    @classmethod
    def extract_jd_required_years(cls, jd_text: str) -> float | None:
        found = [
            float(m.group(1))
            for pat in cls.JD_YEAR_PATTERNS
            for m in pat.finditer(jd_text)
            if 0 < float(m.group(1)) <= 25
        ]
        return max(found) if found else None

    @classmethod
    def extract_resume_experience_years(cls, resume_text: str) -> float | None:
        explicit = [
            float(m.group(1))
            for m in cls.EXP_EXPLICIT_RE.finditer(resume_text)
            if 0 < float(m.group(1)) <= 40
        ]

        current_year = datetime.now().year
        intervals: list[tuple[int, int]] = []
        for line in cls._slice_experience_section(resume_text).splitlines():
            if cls.EDU_LINE_RE.search(line):
                continue
            for m in cls.DATE_RANGE_RE.finditer(line):
                start_y = int(m.group(1))
                end_raw = m.group(2).lower()
                end_y = int(end_raw) if end_raw.isdigit() else current_year
                if start_y <= end_y <= current_year + 1 and (end_y - start_y) <= 35:
                    intervals.append((start_y, end_y))

        interval_years = 0.0
        if intervals:
            merged: list[tuple[int, int]] = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            interval_years = float(sum(max(1, end - start) for start, end in merged))

        candidates = explicit + ([interval_years] if interval_years > 0 else [])
        return round(max(candidates), 1) if candidates else None

    @classmethod
    def _slice_experience_section(cls, text: str) -> str:
        lines = text.splitlines()
        start = next((i for i, ln in enumerate(lines) if cls.EXP_START_RE.match(ln)), None)
        if start is None:
            return text
        end = next((i for i in range(start + 2, len(lines)) if cls.EXP_STOP_RE.match(lines[i])), len(lines))
        return "\n".join(lines[start:end])

    @classmethod
    def extract_education_level(cls, text: str) -> tuple[int, str]:
        for rank, label, pat in cls.EDUCATION_LEVELS:
            if pat.search(text):
                return rank, label
        return 0, "Not Detected"

    # --- TF-IDF Scoring ---

    @staticmethod
    def parse_skills_input(skills_raw: str) -> list[str]:
        seen: dict[str, str] = {}
        for item in re.split(r"[,;\n]+", skills_raw):
            cleaned = re.sub(r"\s+", " ", item).strip()
            if cleaned:
                seen.setdefault(cleaned.lower(), cleaned)
        return list(seen.values())

    def _compute_skill_tfidf_scores(
        self, skills: list[str], resume_texts: list[str]
    ) -> tuple[list[float], list[list[str]], list[list[str]]]:
        compiled = []
        for skill in skills:
            key = skill.lower()
            escaped = re.escape(key).replace(r"\ ", r"\s+")
            pat = re.compile(rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_+#])", re.I)
            compiled.append((skill, key, pat))

        skill_keys = [key for _, key, _ in compiled]
        matched_all, missing_all, matched_keys = [], [], []

        for text in resume_texts:
            matched, missing, keys = [], [], []
            for orig, key, pat in compiled:
                if pat.search(text):
                    matched.append(orig)
                    keys.append(key)
                else:
                    missing.append(orig)
            matched_all.append(matched)
            missing_all.append(missing)
            matched_keys.append(keys)

        try:
            vectorizer = TfidfVectorizer(
                analyzer=lambda tokens: tokens,
                vocabulary=skill_keys,
                binary=True,
                norm="l2",
            )
            matrix = vectorizer.fit_transform([skill_keys] + matched_keys)
            scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten().tolist()
        except ValueError:
            scores = [0.0] * len(resume_texts)

        return scores, matched_all, missing_all

    def _compute_jd_nlp_scores(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        jd_clean = self.preprocess_text_nltk(jd_text)
        if not jd_clean:
            return [0.0] * len(resume_texts)

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
            matrix = vectorizer.fit_transform([jd_clean] + [self.preprocess_text_nltk(t) for t in resume_texts])
            return cosine_similarity(matrix[0:1], matrix[1:]).flatten().tolist()
        except ValueError:
            return [0.0] * len(resume_texts)

    # --- Batch Evaluation ---

    def evaluate_resumes(
        self,
        jd_text: str,
        skills_raw: str,
        pdf_paths: list[str],
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[CandidateResult], dict[str, str]]:
        skills = self.parse_skills_input(skills_raw)
        jd_years = self.extract_jd_required_years(jd_text)
        jd_edu_rank, jd_edu_label = self.extract_education_level(jd_text)

        results: list[CandidateResult] = []
        resume_texts: list[str] = []

        for idx, path in enumerate(pdf_paths, start=1):
            filename = os.path.basename(path)
            if progress_callback:
                progress_callback(f"Parsing PDF {idx}/{len(pdf_paths)}: {filename}")

            raw_text, warning = "", ""
            try:
                raw_text = extract_pdf_text(path)
                if len(raw_text.strip()) < 30:
                    warning = "Very little or no selectable text extracted. This PDF may be a scanned image."
            except Exception as exc:
                warning = f"Failed to read PDF: {exc}"

            exp_years = self.extract_resume_experience_years(raw_text) if raw_text else None
            edu_rank, edu_label = self.extract_education_level(raw_text) if raw_text else (0, "Not Detected")

            exp_score = (
                min(1.0, exp_years / jd_years) if (jd_years and exp_years is not None)
                else (1.0 if jd_years is None and exp_years is not None else 0.0)
            )
            edu_score = (
                min(1.0, edu_rank / jd_edu_rank) if jd_edu_rank > 0
                else (1.0 if edu_rank > 0 else 0.0)
            )

            results.append(CandidateResult(
                filepath=path,
                filename=filename,
                candidate_name=self.extract_candidate_name(raw_text, filename),
                experience_score=exp_score,
                education_score=edu_score,
                detected_experience_years=exp_years,
                detected_education=edu_label,
                warning=warning,
            ))
            resume_texts.append(raw_text)

        if progress_callback:
            progress_callback("Running TF-IDF skill similarity and NLTK JD analysis...")

        skill_scores, matched_lists, missing_lists = self._compute_skill_tfidf_scores(skills, resume_texts)
        jd_scores = self._compute_jd_nlp_scores(jd_text, resume_texts)

        w_skill, w_jd = 0.50, 0.30
        w_exp = 0.12 if jd_years is not None else 0.0
        w_edu = 0.08 if jd_edu_rank > 0 else 0.0
        total_w = w_skill + w_jd + w_exp + w_edu

        for res, s_score, j_score, matched, missing in zip(
            results, skill_scores, jd_scores, matched_lists, missing_lists
        ):
            res.skill_score = s_score
            res.jd_score = j_score
            res.matched_skills = matched
            res.missing_skills = missing
            weighted = (w_skill * s_score + w_jd * j_score + w_exp * res.experience_score + w_edu * res.education_score) / total_w
            res.overall_score = round(weighted * 100.0, 2)

        results.sort(key=lambda r: (r.overall_score, r.skill_score), reverse=True)

        jd_meta = {
            "required_years": f"{jd_years:g}+ yrs" if jd_years is not None else "Not specified",
            "required_education": jd_edu_label if jd_edu_rank > 0 else "Not specified",
            "skill_count": str(len(skills)),
        }
        return results, jd_meta


# =============================================================================
# 4. TKINTER GUI
# =============================================================================

class ResumeScreenerApp:
    COLUMNS = [
        ("rank", "Rank", 52, "center"),
        ("name", "Candidate Name", 180, "w"),
        ("overall", "Overall Score", 100, "center"),
        ("skills", "Skill Match", 90, "center"),
        ("jd", "JD NLP Match", 98, "center"),
        ("matched_cnt", "Skills Hit", 80, "center"),
        ("file", "Resume File", 150, "w"),
    ]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Resume Screener — Corporate Talent Acquisition Suite")
        self.root.geometry("1280x780")
        self.root.minsize(1060, 680)
        self.root.configure(bg="#F1F5F9")

        self.engine: ResumeScreenerEngine | None = None
        self.pdf_files: list[str] = []
        self.all_results: list[CandidateResult] = []
        self.displayed_results: dict[str, CandidateResult] = {}
        self.sort_reverse: dict[str, bool] = {}

        self._configure_styles()
        self._build_ui()

        self.status_var.set("Initializing NLTK resources...")
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _init_engine(self) -> None:
        self.engine = ResumeScreenerEngine()
        self.root.after(0, self.status_var.set, "Ready. Enter a Job Description, required skills, and upload PDF resumes.")

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        configs = {
            "TFrame": {"background": "#F1F5F9"},
            "Card.TFrame": {"background": "#FFFFFF", "relief": "flat"},
            "Header.TFrame": {"background": "#0F172A"},
            "Title.TLabel": {"background": "#0F172A", "foreground": "#FFFFFF", "font": ("Segoe UI", 15, "bold")},
            "Subtitle.TLabel": {"background": "#0F172A", "foreground": "#94A3B8", "font": ("Segoe UI", 9)},
            "Section.TLabel": {"background": "#FFFFFF", "foreground": "#0F172A", "font": ("Segoe UI", 10, "bold")},
            "Muted.TLabel": {"background": "#FFFFFF", "foreground": "#64748B", "font": ("Segoe UI", 9)},
            "Primary.TButton": {"background": "#2563EB", "foreground": "#FFFFFF", "font": ("Segoe UI", 10, "bold"), "padding": (12, 8)},
            "Secondary.TButton": {"background": "#E2E8F0", "foreground": "#0F172A", "font": ("Segoe UI", 9), "padding": (8, 5)},
            "Treeview": {"background": "#FFFFFF", "foreground": "#0F172A", "fieldbackground": "#FFFFFF", "rowheight": 28, "font": ("Segoe UI", 9)},
            "Treeview.Heading": {"background": "#E2E8F0", "foreground": "#0F172A", "font": ("Segoe UI", 9, "bold"), "padding": (6, 6)},
        }
        for name, opts in configs.items():
            style.configure(name, **opts)

        style.map("Primary.TButton", background=[("active", "#1D4ED8"), ("disabled", "#93C5FD")])
        style.map("Secondary.TButton", background=[("active", "#CBD5E1")])
        style.map("Treeview", background=[("selected", "#DBEAFE")], foreground=[("selected", "#1E3A8A")])

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text="AI Resume Screener", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="  |  TF-IDF Skill Matching & NLTK Job Description Analysis", style="Subtitle.TLabel").pack(side=tk.LEFT, pady=(4, 0))
        self.jd_meta_var = tk.StringVar()
        ttk.Label(header, textvariable=self.jd_meta_var, style="Subtitle.TLabel").pack(side=tk.RIGHT)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        left_card = ttk.Frame(main, style="Card.TFrame", padding=14)
        right_card = ttk.Frame(main, style="Card.TFrame", padding=14)
        main.add(left_card, weight=4)
        main.add(right_card, weight=6)

        self._build_left_panel(left_card)
        self._build_right_panel(right_card)

        self.status_var = tk.StringVar(value="Ready.")
        status_bar = tk.Frame(self.root, bg="#E2E8F0", height=28)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status_bar, textvariable=self.status_var, bg="#E2E8F0", fg="#334155", font=("Segoe UI", 9), anchor="w", padx=14).pack(fill=tk.X, pady=3)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="1. Job Description (Plaintext)", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Paste full role responsibilities, qualifications, and experience requirements:", style="Muted.TLabel").pack(anchor="w", pady=(0, 4))

        self.jd_text = scrolledtext.ScrolledText(parent, height=11, wrap=tk.WORD, font=("Segoe UI", 9), bd=1, relief=tk.SOLID, highlightthickness=0)
        self.jd_text.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        ttk.Label(parent, text="2. Required Skills (Comma-Separated)", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Example: Python, SQL, Machine Learning, AWS, Docker, C++, Communication", style="Muted.TLabel").pack(anchor="w", pady=(0, 4))

        self.skills_text = tk.Text(parent, height=3, wrap=tk.WORD, font=("Segoe UI", 9), bd=1, relief=tk.SOLID)
        self.skills_text.pack(fill=tk.X, pady=(0, 12))

        res_hdr = ttk.Frame(parent, style="Card.TFrame")
        res_hdr.pack(fill=tk.X)
        ttk.Label(res_hdr, text="3. Candidate Resumes (PDF)", style="Section.TLabel").pack(side=tk.LEFT)
        self.pdf_count_var = tk.StringVar(value="0 files loaded")
        ttk.Label(res_hdr, textvariable=self.pdf_count_var, style="Muted.TLabel").pack(side=tk.RIGHT)

        btn_row = ttk.Frame(parent, style="Card.TFrame")
        btn_row.pack(fill=tk.X, pady=6)
        for label, cmd in [
            ("+ Add PDFs", self._add_pdfs),
            ("+ Add Folder", self._add_folder),
            ("Remove Selected", self._remove_selected_pdf),
            ("Clear All", self._clear_pdfs),
        ]:
            ttk.Button(btn_row, text=label, style="Secondary.TButton", command=cmd).pack(side=tk.LEFT, padx=(0, 6))

        list_frame = tk.Frame(parent, bd=1, relief=tk.SOLID)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.pdf_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, font=("Segoe UI", 9), bd=0, highlightthickness=0, height=7)
        pdf_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pdf_listbox.yview)
        self.pdf_listbox.configure(yscrollcommand=pdf_scroll.set)
        self.pdf_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pdf_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.screen_btn = ttk.Button(parent, text="Screen & Rank Candidates", style="Primary.TButton", command=self._start_screening)
        self.screen_btn.pack(fill=tk.X)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top_bar = ttk.Frame(parent, style="Card.TFrame")
        top_bar.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(top_bar, text="Ranked Candidates", style="Section.TLabel").pack(side=tk.LEFT)

        search_box = ttk.Frame(top_bar, style="Card.TFrame")
        search_box.pack(side=tk.RIGHT)
        ttk.Label(search_box, text="Search Candidate Name:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh_results_table())
        ttk.Entry(search_box, textvariable=self.search_var, width=26, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(search_box, text="Clear", style="Secondary.TButton", command=lambda: self.search_var.set("")).pack(side=tk.LEFT)

        self.match_count_var = tk.StringVar(value="No screening results yet.")
        ttk.Label(parent, textvariable=self.match_count_var, style="Muted.TLabel").pack(anchor="w", pady=(0, 6))

        table_frame = ttk.Frame(parent, style="Card.TFrame")
        table_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(table_frame, columns=[c[0] for c in self.COLUMNS], show="headings", selectmode="browse")
        for col, text, width, anchor in self.COLUMNS:
            self.tree.heading(col, text=text, command=lambda c=col: self._sort_tree(c))
            self.tree.column(col, width=width, anchor=anchor)

        for tag, bg in (("high", "#ECFDF5"), ("medium", "#FFFBEB"), ("low", "#FEF2F2")):
            self.tree.tag_configure(tag, background=bg)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._show_candidate_details(self._selected_result()))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        detail_hdr = ttk.Frame(parent, style="Card.TFrame")
        detail_hdr.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(detail_hdr, text="Candidate Evaluation Breakdown", style="Section.TLabel").pack(side=tk.LEFT)
        self.open_pdf_btn = ttk.Button(detail_hdr, text="Open Resume PDF", style="Secondary.TButton", command=self._open_selected_pdf, state=tk.DISABLED)
        self.open_pdf_btn.pack(side=tk.RIGHT)

        self.detail_text = scrolledtext.ScrolledText(parent, height=11, wrap=tk.WORD, font=("Segoe UI", 9), bg="#F8FAFC", bd=1, relief=tk.SOLID, state=tk.DISABLED)
        self.detail_text.pack(fill=tk.BOTH)
        self.detail_text.tag_configure("heading", font=("Segoe UI", 10, "bold"), foreground="#0F172A")
        self.detail_text.tag_configure("good", foreground="#15803D", font=("Segoe UI", 9, "bold"))
        self.detail_text.tag_configure("bad", foreground="#B91C1C", font=("Segoe UI", 9, "bold"))
        self.detail_text.tag_configure("warn", foreground="#B45309", font=("Segoe UI", 9, "italic"))

    # --- File Management ---

    def _register_pdfs(self, paths: list[str], status_fmt: str) -> None:
        added = 0
        for path in paths:
            if path not in self.pdf_files:
                self.pdf_files.append(path)
                self.pdf_listbox.insert(tk.END, os.path.basename(path))
                added += 1
        self._update_pdf_count()
        if added:
            self.status_var.set(status_fmt.format(added=added))

    def _add_pdfs(self) -> None:
        paths = filedialog.askopenfilenames(title="Select Candidate Resume PDFs", filetypes=[("PDF Files", "*.pdf")])
        self._register_pdfs(list(paths), "Added {added} resume(s).")

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Folder Containing Resume PDFs")
        if not folder:
            return
        pdfs = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(".pdf")]
        self._register_pdfs(pdfs, "Loaded {added} PDF(s) from folder.")

    def _remove_selected_pdf(self) -> None:
        for idx in reversed(self.pdf_listbox.curselection()):
            self.pdf_listbox.delete(idx)
            del self.pdf_files[idx]
        self._update_pdf_count()

    def _clear_pdfs(self) -> None:
        self.pdf_files.clear()
        self.pdf_listbox.delete(0, tk.END)
        self._update_pdf_count()

    def _update_pdf_count(self) -> None:
        count = len(self.pdf_files)
        self.pdf_count_var.set(f"{count} file{'s' if count != 1 else ''} loaded")

    # --- Screening Execution ---

    def _start_screening(self) -> None:
        engine = self.engine
        if engine is None:
            messagebox.showinfo("Please Wait", "NLP engine is still initializing. Try again in a moment.")
            return

        jd = self.jd_text.get("1.0", tk.END).strip()
        skills = self.skills_text.get("1.0", tk.END).strip()

        if not jd:
            messagebox.showwarning("Missing Job Description", "Please enter the plaintext Job Description.")
            return
        if not skills:
            messagebox.showwarning("Missing Required Skills", "Please enter at least one required skill.")
            return
        if not self.pdf_files:
            messagebox.showwarning("No Resumes Loaded", "Please add at least one candidate PDF resume.")
            return

        self.screen_btn.configure(state=tk.DISABLED)
        self.status_var.set("Starting resume screening...")

        def worker() -> None:
            try:
                results, jd_meta = engine.evaluate_resumes(
                    jd_text=jd,
                    skills_raw=skills,
                    pdf_paths=list(self.pdf_files),
                    progress_callback=lambda msg: self.root.after(0, self.status_var.set, msg),
                )
                self.root.after(0, self._on_screening_complete, results, jd_meta)
            except Exception as exc:
                self.root.after(0, self._on_screening_error, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_screening_complete(self, results: list[CandidateResult], jd_meta: dict[str, str]) -> None:
        self.all_results = results
        self.screen_btn.configure(state=tk.NORMAL)
        self.jd_meta_var.set(
            f"JD Criteria Detected  |  Skills: {jd_meta['skill_count']}  •  "
            f"Exp: {jd_meta['required_years']}  •  Edu: {jd_meta['required_education']}"
        )
        self._refresh_results_table()
        self.status_var.set(f"Screening complete. Ranked {len(results)} candidate(s).")

    def _on_screening_error(self, error_msg: str) -> None:
        self.screen_btn.configure(state=tk.NORMAL)
        self.status_var.set("Error during screening.")
        messagebox.showerror("Screening Error", f"An error occurred while screening resumes:\n{error_msg}")

    # --- Results Table & Details ---

    def _refresh_results_table(self) -> None:
        query = self.search_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        self.displayed_results.clear()

        if not self.all_results:
            self.match_count_var.set("No screening results yet.")
            return

        for rank, res in enumerate(self.all_results, start=1):
            if query and query not in res.candidate_name.lower():
                continue

            total_skills = len(res.matched_skills) + len(res.missing_skills)
            tag = "high" if res.overall_score >= 65 else ("medium" if res.overall_score >= 40 else "low")
            item_id = f"cand_{rank}"
            self.displayed_results[item_id] = res

            self.tree.insert(
                "",
                tk.END,
                iid=item_id,
                values=(
                    f"#{rank}",
                    res.candidate_name,
                    f"{res.overall_score:.1f}%",
                    f"{res.skill_score * 100:.1f}%",
                    f"{res.jd_score * 100:.1f}%",
                    f"{len(res.matched_skills)}/{total_skills}",
                    res.filename,
                ),
                tags=(tag,),
            )

        shown = len(self.displayed_results)
        if query:
            self.match_count_var.set(f"Showing {shown} of {len(self.all_results)} candidates matching '{self.search_var.get().strip()}'.")
        else:
            self.match_count_var.set(f"Showing all {shown} ranked candidates (sorted by Overall Score).")

        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self._show_candidate_details(self.displayed_results[children[0]])
        else:
            self._show_candidate_details(None)

    def _sort_tree(self, col: str) -> None:
        reverse = self.sort_reverse.get(col, False)
        self.sort_reverse[col] = not reverse

        def sort_key(item_id: str):
            val = self.tree.set(item_id, col)
            if col == "rank":
                return int(val.lstrip("#"))
            if col in ("overall", "skills", "jd"):
                return float(val.rstrip("%"))
            if col == "matched_cnt":
                return int(val.split("/")[0])
            return val.lower()

        for idx, iid in enumerate(sorted(self.tree.get_children(""), key=sort_key, reverse=reverse)):
            self.tree.move(iid, "", idx)

    def _selected_result(self) -> CandidateResult | None:
        selection = self.tree.selection()
        return self.displayed_results.get(selection[0]) if selection else None

    def _show_candidate_details(self, res: CandidateResult | None) -> None:
        self.open_pdf_btn.configure(state=tk.NORMAL if res else tk.DISABLED)
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)

        if res:
            exp = f"{res.detected_experience_years:g} yrs" if res.detected_experience_years is not None else "Not Detected"
            entries = [
                (f"{res.candidate_name}  ", "heading"),
                (f"({res.filename})\n", None),
                (
                    f"Overall Fit: {res.overall_score:.1f}%   |   "
                    f"Skill Match (TF-IDF Cosine): {res.skill_score * 100:.1f}%   |   "
                    f"JD Alignment (NLTK): {res.jd_score * 100:.1f}%\n"
                    f"Detected Experience: {exp} ({res.experience_score * 100:.0f}% fit)   |   "
                    f"Detected Education: {res.detected_education} ({res.education_score * 100:.0f}% fit)\n\n",
                    None,
                ),
                ("Matched Required Skills: ", "good"),
                (f"{', '.join(res.matched_skills) or 'None'}\n", None),
                ("Missing Required Skills: ", "bad"),
                (f"{', '.join(res.missing_skills) or 'None — all required skills present!'}\n", None),
            ]
            if res.warning:
                entries.append((f"\nNote: {res.warning}\n", "warn"))

            for text, tag in entries:
                if tag:
                    self.detail_text.insert(tk.END, text, tag)
                else:
                    self.detail_text.insert(tk.END, text)

        self.detail_text.configure(state=tk.DISABLED)

    def _open_selected_pdf(self) -> None:
        res = self._selected_result()
        if res and (pdf_path := Path(res.filepath)).exists():
            webbrowser.open(pdf_path.resolve().as_uri())


if __name__ == "__main__":
    root = tk.Tk()
    ResumeScreenerApp(root)
    root.mainloop()