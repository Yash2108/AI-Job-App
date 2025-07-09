from dateutil import parser
from datetime import datetime
import re
import pymupdf

def extract_text_from_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
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

resume_path = 'resume.pdf'
extracted_resume = extract_text_from_pdf(resume_path)
work_exp = extract_work_experience_section(extracted_resume)
projects = extract_projects_section(extracted_resume)
education = extract_education_section(extracted_resume)
skills = extract_skills_section(extracted_resume)
years_of_exp = extract_experience_years(work_exp)

print(years_of_exp)