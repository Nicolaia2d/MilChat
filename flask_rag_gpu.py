from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import transformers

# === Start Flask ===
app = Flask(__name__)

# === Les FAISS index + metadata ===
print("📌 Leser FAISS index og metadata...")
index = faiss.read_index("faiss_index.idx")
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
print(f"✅ Index: {index.ntotal} embeddings lastet.")

# === Initialiser embedder ===
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Aktiver detaljert transformers-logg ===
transformers.logging.set_verbosity_info()

# === GPU sjekk ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Torch CUDA tilgjengelig? {torch.cuda.is_available()}")
print(f"📌 Enhet: {device}")
if device == "cuda":
    print(f"📌 GPU-navn: {torch.cuda.get_device_name(0)}")

# === Last LLM ===
model_name = "NorwAI/NorwAI-Mistral-7B-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "10GB", "cpu": "30GB"},
    low_cpu_mem_usage=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# === Last chunks.txt ===
chunks = []
with open("chunks.txt", "r", encoding="utf-8") as f:
    for line in f:
        page, text = line.strip().split("||")
        chunks.append({"page": int(page), "text": text})

# === Hjemmeside ===
@app.route("/", methods=["GET"])
def home():
    return "<h2>✅ Flask RAG på GPU kjører! Bruk POST /ask med JSON {'question': '...'}</h2>"

# === Spørring ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Du må sende {'question': '...'}"}), 400

    # 1) Embed spørsmål
    q_embed = embedder.encode([question]).astype('float32')

    # 2) FAISS-søk
    D, I = index.search(q_embed, k=1)
    top_idx = I[0][0]
    top_page = metadata[top_idx]["page"]

    # 3) Finn chunk
    chunk = next((c for c in chunks if c["page"] == top_page), None)
    if not chunk:
        return jsonify({"error": "Fant ikke chunk-tekst."}), 500

    # 4) Bygg prompt
    prompt = f"""
    Spørsmål: {question}

    Relevant informasjon:
    - Side {top_page}: {chunk['text']} (UD-2.1, s. {top_page})

    Svar kort, med kilde.
    """

    # 5) Generer svar
    outputs = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.0
    )

    answer = outputs[0]["generated_text"]

    return jsonify({
        "answer": answer,
        "source": f"UD-2.1, side {top_page}"
    })

# === Kjør server ===
if __name__ == "__main__":
    app.run(debug=True)
