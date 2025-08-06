
from preprocess_utils import *
from pos_tagger import *
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# from dotenv import load_dotenv
# import os

# load_dotenv()

# HF_TOKEN_READ = os.getenv("HF_TOKEN_READ")
# HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")


SYSTEM_PROMPT='''You are a resume scoring assistant. Score the following resume from 0 to 100 based on how well it matches the job description. 
Input format:
Description:
<job description>

Resume:
<resume>

Only respond in the following format:
Score: <number>

Do not include any other text.'''

def format_chat(system_prompt, user_input):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

def use_chat_format(user_input, model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
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

# Use a pipeline as a high-level helper

def use_text_generation_pipeline(user_input, model = "openai-community/gpt2"):
    pipe = pipeline("text-generation", model = model)
    output = pipe(user_input)
    return output

def score_resume(extracted_resume, extracted_jd):
    # input_text = f'''
    # Description:
    # {extracted_jd}

    # Resume:
    # {extracted_resume}
    # '''

    input_text = '''You are a resume scoring assistant. Score the following resume from 0 to 100 based on how well it matches the job description. Use the following weights:

Skill Match: 40%
Responsibility Match: 30%
Experience Match: 20%
Education Match: 10%

Input:
Description:
{extracted_jd}

Resume:
{extracted_resume}

Only respond in the following format:
Score: <number>
Reason: <your analysis broken down by each scoring category>

Do not include any other text.
'''
    # output = use_chat_format(input_text, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")
    output = use_chat_format(input_text)
    return output

if __name__ == "__main__":

    resume_path = 'resume.pdf'
    job_description_path = 'job_description.txt'

    extracted_resume = extract_text_from_pdf(resume_path)
    extracted_jd = read_from_txt(job_description_path)

    print(score_resume(extracted_resume, extracted_jd))

