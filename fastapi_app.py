from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "NorwAI/NorwAI-Mistral-7B-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

@app.post("/generate")
def generate(prompt: Prompt):
    outputs = generator(
        prompt.prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )
    result = outputs[0]["generated_text"]
    return {"generated_text": result}
