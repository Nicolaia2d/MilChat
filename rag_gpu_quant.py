from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import transformers

app = Flask(__name__)

# === Leser FAISS index + metadata ===
print("📌 Leser FAISS index og metadata...")
index = faiss.read_index("faiss_index.idx")
with open("faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
print(f"✅ Index: {index.ntotal} embeddings lastet.")

# === Initialiser embedder ===
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === Logging for Transformers ===
transformers.logging.set_verbosity_info()

# === CUDA-status ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Torch CUDA tilgjengelig? {torch.cuda.is_available()}")
print(f"📌 Enhet: {device}")
if device == "cuda":
    print(f"📌 GPU-navn: {torch.cuda.get_device_name(0)}")

# === Modellnavn ===
model_name = "NorwAI/NorwAI-Mistral-7B-instruct"

# === Bruk bitsandbytes (4-bit quantization) ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# === Last inn tokenizer og LLM ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# === Pipeline ===
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# === Last chunks ===
chunks = []
with open("chunks.txt", "r", encoding="utf-8") as f:
    for line in f:
        page, text = line.strip().split("||")
        chunks.append({"page": int(page), "text": text})

@app.route("/", methods=["GET"])
def home():
    return "<h2>✅ Flask RAG med 4-bit quantization kjører! POST /ask med JSON {'question': '...'}</h2>"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Du må sende {'question': '...'}"}), 400

    # 1) Embed spørsmålet
    q_embed = embedder.encode([question]).astype('float32')

    # 2) FAISS søk
    D, I = index.search(q_embed, k=3)
    top_idx = I[0][0]
    top_page = metadata[top_idx]["page"]

    # 3) Hent chunk
    chunk = next((c for c in chunks if c["page"] == top_page), None)
    if not chunk:
        return jsonify({"error": "Fant ikke chunk-tekst."}), 500

    # 4) Sterk prompt
    prompt = f"""Du er en militær AI-assistent.
Bruk kun teksten i KONTEKST. Ikke gjett eller finn opp.
Svar ordrett med punktene fra teksten hvis mulig.
Legg alltid til kilde: UD-2.1, side {top_page}.

KONTEKST:
{chunk['text']}

SPØRSMÅL: {question}

SVAR:
"""

    # 5) Generer
    outputs = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    raw_output = outputs[0]["generated_text"]

    if "SVAR:" in raw_output:
        answer = raw_output.split("SVAR:")[1].strip()
    else:
        answer = raw_output.strip()

    return jsonify({
        "answer": answer,
        "source": f"UD-2.1, side {top_page}"
    })


if __name__ == "__main__":
    app.run(debug=True)
