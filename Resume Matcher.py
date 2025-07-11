import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from preprocess_utils import *
from pos_tagger import *

model = SentenceTransformer("joeportnoy/resume-match-ml")

'''

1. Read Resume
2. Extract information from Resume
    2.1 Years of Exp
    2.2 Skills
    2.3 Qualifications
    2.4 Roles and Experience Details
    2.5 Education

3. Extract info from Job Description
    3.1 Skills
    3.2 Years of Experience required (Qualifications)
    3.3 Role Description

4. Preprocess
    4.1 Lowercase
    4.2 Map Abbreviations
    4.3 remove stopwords
    4.4 Lemmatization

    
5. Match the information to get a score

------------------------------------------------------------------------------------------------
Category	                   | Weight (%)	|    Method
-------------------------------+------------+---------------------------------------------------
Skill match                    |     40%	|        Keyword match
Responsibility match	       |     30%	|        Sentence similarity
Experience match	           |     20%	|        NLP + rule-based: years, roles, industries 
Education match	               |     10%	|        Degree-level & field matching
-------------------------------------------------------------------------------------------------

'''
def get_embedding(text):
    return model.encode([text])[0]

def similarity_score(text1, text2):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    return float(cosine_similarity([emb1], [emb2])[0][0])

# -----------------------------
# 4. Core Scoring Engine
# -----------------------------
def score_resume(resume_text, job_text):
    resume_text = preprocess_text(resume_text)
    job_text = preprocess_text(job_text)

    work_exp = extract_work_experience_section(resume_text)
    projects = extract_projects_section(resume_text)
    education = extract_education_section(resume_text)
    skills = extract_skills_section(resume_text)
    years_of_exp = extract_experience_years(work_exp)
    
    keywords_resume = []
    for section in [work_exp, projects, education, skills]:
        keywords_resume.append(extract_pos_keywords(section))

    keywords_resume = set([ word for words in keywords_resume for word in words ])

    keywords_jd = extract_pos_keywords(job_text)

    # Skill Match
    matched_skills = list(keywords_resume & keywords_jd)
    missing_skills = list(keywords_jd - keywords_resume)
    skill_score = len(matched_skills) / max(len(keywords_jd), 1)

    # TODO: Figure out how to score experience (For now, set to boolean)
    # Experience Match (Very basic check for now)
    jd_experience_required = extract_required_years(extracted_jd)
    experience_score = 1.0 if jd_experience_required <= years_of_exp else 0.0

    # TODO: Extract most recent education from resume 
    # TODO: Identify education required and match it with the most recent education extracted from resume
    # Education Match (basic)
    # Education Match
    required_degree, preferred_degree = extract_degree_requirements(job_text)
    candidate_degree = extract_most_recent_education(resume_text)

    degree_levels = ["associate", "bachelor", "mba", "master", "doctorate", "phd"]

    def degree_rank(degree):
        return degree_levels.index(degree) if degree in degree_levels else -1

    education_score = 0.0
    education_match = False

    if candidate_degree:
        if required_degree and degree_rank(candidate_degree) >= degree_rank(required_degree):
            education_score = 1.0
            education_match = True
        elif preferred_degree and degree_rank(candidate_degree) >= degree_rank(preferred_degree):
            education_score = 0.5
            education_match = True

    # Overall Similarity
    all_sections = work_exp + projects + education + skills
    overall_similarity = similarity_score(all_sections, job_text)

    # Weighted Score
    score = (
        0.4 * skill_score +
        0.3 * overall_similarity +
        0.2 * experience_score +
        0.1 * education_score
    ) * 100

    return {
        "score": round(score, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "experience_match": experience_score,
        "education_match": education_match,
        "notes": f"Matched {len(matched_skills)} of {len(keywords_jd)} key skills."
    }

# -----------------------------
# 5. Example Usage
# -----------------------------
if __name__ == "__main__":

    resume_path = 'resume.pdf'
    job_description_path = 'job_description.txt'

    extracted_resume = extract_text_from_pdf(resume_path)
    extracted_jd = read_from_txt(job_description_path)

    # result = score_resume(extracted_resume, extracted_jd)
    # print(json.dumps(result, indent=2))
    # print(extract_required_years(extracted_jd))
    print(score_resume(extracted_resume, extracted_jd))
