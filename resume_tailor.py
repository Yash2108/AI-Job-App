from preprocess_utils import *
from huggingface_hub import InferenceClient
import json
from dotenv import load_dotenv
import os
from datetime import datetime
load_dotenv()

HF_TOKEN_READ = os.getenv("HF_TOKEN_READ")
# HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")

SYSTEM_PROMPT = '''You are an expert at tailoring resumes. Given a job description and a resume bullet point, make the necessary changes to make sure the bullet point caters to some part of the job description.

Input format:
Description:
<job description>

Resume Bullet point:
<point>

Output:
Write down 1 bullet point with atmost 100 characters.

Do not include any other text.
'''


def format_chat(system_prompt, user_input):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

def inference_llama_3b(user_input):
    client = InferenceClient(
        provider="novita",
        api_key=HF_TOKEN_READ,
    )
    messages = format_chat(SYSTEM_PROMPT, user_input)

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=1024,
        n=3
    )
    output = [choice.message.content for choice in completion.choices ]
    return output

def tailor_bullet_point(resume_bullet_point, extracted_jd):
    input_text = f'''
    Description:
    {extracted_jd}

    Resume Bullet point:
    {resume_bullet_point}
    '''
    output = inference_llama_3b(input_text)

    return output


if __name__ == "__main__":

    resume_path = 'resume.pdf'
    job_description_path = 'job_description.txt'

    resume_bullet_point = 'Building a personalized recommender system using Python (Pandas, Scikit-learn) and integrating it into a fullstack application with a Spring Boot backend and Angular frontend'
    extracted_jd = read_from_txt(job_description_path)

    options = tailor_bullet_point(resume_bullet_point, extracted_jd)

    for option in options:
        print(option)
