from preprocess_utils import read_from_txt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("nakamoto-yama/t5-resume-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("nakamoto-yama/t5-resume-generation")


bullet_point = '''Boosted team onboarding efficiency by 50% by building an in-house ATS using Python, \
SpaCy and SciKit-Learn to parse resumes and recommend candidates to teams via customized ML-based matching system'''
job_description_path = 'job_description.txt'

extracted_jd = read_from_txt(job_description_path)


prompt = f'''
You are an expert resume writer. I will share with you a bullet point and a job description for which I am applying. 
Tailor this bullet point to fit the job.
Here is the bullet point:
{bullet_point}

Here is the job description:
{extracted_jd}
'''
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
output_ids = model.generate(input_ids, max_length=512)

generated_point = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_point)