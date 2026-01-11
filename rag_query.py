import faiss, pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# === Last index og metadata ===
index = faiss.read_index("faiss_index.idx")
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# === Last embedding-modell ===
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Last LLM ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "NorwAI/NorwAI-Mistral-7B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Søk + prompt ===
question = "Hva er maksimal rekkevidde på AG3?"
q_embed = embedder.encode([question]).astype('float32')

D, I = index.search(q_embed, k=1)
top = I[0][0]
page = metadata[top]["page"]

# Du må hente chunk-text: f.eks. fra fil/database.
chunk_text = "Eksempel: AG3 har en effektiv rekkevidde på 400 meter."

prompt = f"""
Spørsmål: {question}

Relevant informasjon:
- Side {page}: {chunk_text} (UD-2.1, s. {page})

Svar kort, med kilde.
"""

outputs = generator(
    prompt,
    max_new_tokens=50,
    do_sample=False,
    temperature=0.0
)

print("\n=== Svar ===")
print(outputs[0]["generated_text"])
