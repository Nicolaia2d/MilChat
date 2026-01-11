from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Flask(__name__)

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

@app.route("/", methods=["GET"])
def home():
    return "<h2>✅ Flask kjører! Send POST til /generate med JSON prompt.</h2>"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    outputs = generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )
    result = outputs[0]["generated_text"]
    return jsonify({"generated_text": result})

if __name__ == "__main__":
    app.run(debug=True)
