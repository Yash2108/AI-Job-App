from preprocess_utils import *
from huggingface_hub import InferenceClient
import json
from dotenv import load_dotenv
import os
from datetime import datetime
load_dotenv()

HF_TOKEN_READ = os.getenv("HF_TOKEN_READ")
# HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")


SYSTEM_PROMPT='''You are a resume scoring assistant. Score the following resume on each parameter from the following metric from 0 to 100:

Skill Match: Look for the skills asked for in the job description and check if they are present in the resume
Responsibility match: For the responsibilities mentioned in the job description and check if the resume has mentioned them
Experience match: Count the number of years and domain required in the job description and check if the resume matches the exact requirement
Education match: Look for the education required and check the resume for a match

Input format:
Description:
<job description>

Resume:
<resume>

Output in JSON format as follows:

{
  "Skill Match": [<score>, <reason>],
  "Responsibility Match": [<score>, <reason>],
  "Experience Match": [<score>, <reason>],
  "Education Match": [<score>, <reason>],
}

Do not include any other text.'''


def save_json(data, filename=None):
    folder_path = "./output_jsons"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if not filename:
        filename = datetime.now().strftime("%Y%m%d-%H%M%S")+".json"
        file_path = os.path.join(folder_path, filename)
    else:
        file_path = filename
        
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)
    
    return file_path

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
        max_tokens=1024
    )

    return completion.choices[0].message

def score_resume(extracted_resume, extracted_jd):
    input_text = f'''
    Description:
    {extracted_jd}

    Resume:
    {extracted_resume}
    '''
    output = inference_llama_3b(input_text)

    match = re.search(r'\{[\s\S]*?\}', output.content)
    if match:
        json_text = match.group(0)
        try:
            return save_json(data=json.loads(json_text))
        except Exception as e:
            print("Couldn't decode JSON. Error:", e)
            print("original output:", json_text)

    else:
        print("No JSON found in output:", json_text)
    return None


if __name__ == "__main__":

    resume_path = 'resume.pdf'
    job_description_path = 'job_description.txt'

    extracted_resume = extract_text_from_pdf(resume_path)
    extracted_jd = read_from_txt(job_description_path)

    filename = score_resume(extracted_resume, extracted_jd)

    if filename:
        with open(filename, 'r') as file:
            data = json.load(file)

        skill_match = data["Skill Match"][0]
        responsibility_match = data["Responsibility Match"][0]
        experience_match = data["Experience Match"][0]
        education_match = data["Education Match"][0]

        data['total_score'] = (skill_match * 0.4) +(responsibility_match * 0.3) + (experience_match * 0.2) + (education_match * 0.1)

        save_json(data=data, filename=filename)
