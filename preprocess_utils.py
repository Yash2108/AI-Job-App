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

def extract_degree_requirements(job_description):
    # Lowercase for simpler matching
    jd = job_description.lower()

    # Education patterns
    degrees = [
        "phd", "doctorate", "m.d.", "md", "dphil",
        "master", "m.sc", "ms", "mba", "m.a.",
        "bachelor", "b.sc", "bs", "b.a.", "beng",
        "associate", "diploma", "high school", "ged"
    ]
    degree_pattern = '|'.join(degrees)

    # Sentence patterns
    required_patterns = [
        r"(required|must have|necessar(y|ily))[^.]*(" + degree_pattern + r")[^.]*\."
    ]
    preferred_patterns = [
        r"(preferred|ideally|nice to have|would be a plus)[^.]*(" + degree_pattern + r")[^.]*\."
    ]

    required = []
    preferred = []

    for pattern in required_patterns:
        matches = re.findall(pattern, jd)
        required.extend([" ".join(match) for match in matches])

    for pattern in preferred_patterns:
        matches = re.findall(pattern, jd)
        preferred.extend([" ".join(match) for match in matches])

    # Cleanup to get full sentences instead of just tuples
    required_sentences = [m.group(0) for m in re.finditer(required_patterns[0], jd)]
    preferred_sentences = [m.group(0) for m in re.finditer(preferred_patterns[0], jd)]

    return {
        "required_education": required_sentences or ["Not found"],
        "preferred_education": preferred_sentences or ["Not found"]
    }


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



# List of education levels in order of hierarchy
education_levels = [
    "PhD", "Doctorate", "M.D.", "MD", "DPhil",
    "Master", "M.Sc", "M.S.", "MEng", "MBA", "M.A.",
    "Bachelor", "B.Sc", "B.S.", "B.A.", "BEng", "Undergraduate",
    "Associate", "Diploma", "High School", "GED"
]

# Normalize for matching
education_rank = {edu.lower(): rank for rank, edu in enumerate(education_levels)}

def extract_highest_education(resume_text):
    found_levels = []

    for edu in education_levels:
        pattern = re.compile(rf'\b{edu}\b', re.IGNORECASE)
        if re.search(pattern, resume_text):
            found_levels.append(edu.lower())

    if not found_levels:
        return "No education level found."

    # Get the highest-ranking education found
    highest = min(found_levels, key=lambda x: education_rank.get(x, float('inf')))
    return highest

# Example usage
resume = """
John Doe is a data scientist with experience in AI and machine learning. 
He completed his Bachelor of Science in Computer Engineering and recently earned a Master of Science in Data Science from Stanford University.
"""

print(extract_highest_education(resume))