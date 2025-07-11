from dateutil import parser
from datetime import datetime
import re
import pymupdf
from string import punctuation

def extract_text_from_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def read_from_txt(file_path):
    text = ''
    with open(file_path, 'r') as openfile:
        for line in openfile.readlines():
            text+=line
    return text

def extract_section(text, section_name, stop_sections):
    pattern = rf"(?i)({section_name})\s*(.*?)\s*(?={'|'.join(stop_sections)}|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2).strip() if match else ""

def extract_work_experience_section(text):
    return extract_section(
        text,
        "experience|work experience|professional experience|employment|work history",
        ["education", "certifications", "skills", "projects"]
    )

def extract_projects_section(text):
    return extract_section(
        text,
        "projects|personal projects|academic projects",
        ["education", "certifications", "skills", "experience"]
    )

def extract_education_section(text):
    return extract_section(
        text,
        "education|academic background|qualifications",
        ["experience", "skills", "projects"]
    ) 

def extract_skills_section(text):
    return extract_section(
        text,
        "skills|technical skills|key skills",
        ["education", "experience", "certifications", "projects"]
    )

def extract_experience_years(text):
    text = text.lower()
    date_pattern = r"[jan|feb|mar|apr|may|jun|june|jul|july|aug|sep|sept|oct|nov|dec][a-z]*\s*\d{4}"
    matches = re.findall(date_pattern, text)
    dates = re.findall(rf"{date_pattern}", text)

    # Extract date pairs (assume they appear in order: start -> end)
    raw_ranges = re.findall(
        rf"({date_pattern})\s*(?:-|–|to)\s*({date_pattern}|present)", text
    )
    total_months = 0
    for start_str, end_str in raw_ranges:
        try:
            start = parser.parse(start_str)
            end = datetime.today() if 'present' in end_str else parser.parse(end_str)
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if months > 0:
                total_months += months
        except Exception as e:
            continue

    years = round(total_months / 12.0, 1)
    return years

def extract_required_years(job_text):
    pattern = r"(?:at\s+least|minimum|require(?:d)?|with)\s+(\d+(?:\.\d+)?)\+?\s+(?:years?|yrs?)\s+(?:of\s+)?experience"
    match = re.search(pattern, job_text, re.IGNORECASE)
    return float(match.group(1)) if match else None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[\n\r]+", " ", text)
    text = text.translate(str.maketrans('', '', punctuation))
    return text


def extract_degree_requirements(job_text):
    job_text = job_text.lower()

    degree_levels = ["phd", "doctorate", "master", "bachelor", "associate", "mba"]
    required = None
    preferred = None

    for level in degree_levels:
        pattern_required = rf"(?:require(?:d)?|must have|need(?:ed)?)[^\\.\\n]*\b{level}"
        pattern_preferred = rf"(?:prefer(?:red)?|nice to have|ideal[^\\.\\n]*\b{level})"

        if re.search(pattern_required, job_text):
            required = level
        if re.search(pattern_preferred, job_text):
            preferred = level

    return required, preferred

def extract_most_recent_education(resume_text):
    education_text = extract_education_section(resume_text).lower()

    degree_levels = ["phd", "doctorate", "master", "mba", "bachelor", "associate"]
    date_pattern = r"(19|20)\d{2}"

    # Find degrees with dates (if any)
    degree_entries = []
    for level in degree_levels:
        pattern = rf"\b({level})\b.*?({date_pattern})?"
        matches = re.findall(pattern, education_text)
        for match in matches:
            year = int(match[1]) if match[1] else 0
            degree_entries.append((level, year))

    # Sort by year descending
    if degree_entries:
        degree_entries.sort(key=lambda x: (-x[1], degree_levels.index(x[0])))
        return degree_entries[0][0]  # most recent/highest degree

    return None
