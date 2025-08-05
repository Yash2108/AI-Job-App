
from preprocess_utils import *
from pos_tagger import *

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch


SYSTEM_PROMPT='''You are a resume scoring assistant. Score the following resume from 0 to 100 based on how well it matches the job description. Use the following weights:

Skill Match: 40%
Responsibility Match: 30%
Experience Match: 20%
Education Match: 10%

Input format:
Description:
<job description>

Resume:
<resume>

Only respond in the following format:
Score: <number>
Reason: <your analysis broken down by each scoring category>

Do not include any other text.'''

def format_chat(system_prompt, user_input):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

def parse_input(user_input, model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    messages = format_chat(SYSTEM_PROMPT, user_input)
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        input_ids,
        # max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id  # stop generation
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def score_resume(extracted_resume, extracted_jd):
    input_text = f'''
    Description:
    {extracted_jd}

    Resume:
    {extracted_resume}
    '''

    return parse_input(input_text)

if __name__ == "__main__":

    resume_path = 'resume.pdf'
    job_description_path = 'job_description.txt'

    extracted_resume = extract_text_from_pdf(resume_path)
    extracted_jd = read_from_txt(job_description_path)

    print(score_resume(extracted_resume, extracted_jd))

