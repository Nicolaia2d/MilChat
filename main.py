from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sys

# 1️⃣ Sjekk om CUDA er tilgjengelig
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Bruker device: {device}")

# 2️⃣ Modellvalg: stor for GPU, liten for CPU
if device == "cuda":
    model_name = "NorwAI/NorwAI-Mistral-7B-instruct"
else:
    print("⚠️ CUDA er ikke tilgjengelig. Bytter til liten modell (GPT-2) for CPU.")
    model_name = "gpt2"

# 3️⃣ Last tokenizer og modell
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
except Exception as e:
    print(f"❌ Klarte ikke laste modellen: {e}")
    sys.exit(1)


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)



prompt = "Hva er hovedstaden i Norge?"

try:
    outputs = generator(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    result = outputs[0]["generated_text"]
except Exception as e:
    print(f"❌ Feil under generering: {e}")
    sys.exit(1)


print("\n📦 Svar fra modellen:")
print(result)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("Svar lagret i output.txt")
