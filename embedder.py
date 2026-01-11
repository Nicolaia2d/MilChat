import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def pdf_to_chunks_per_page(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            chunks.append((page_num, text))
    print(f"✅ PDF-en ble splittet i {len(chunks)} side-chunks.")
    return chunks

def create_faiss_index(chunks, embedding_model):
    texts = [text for _, text in chunks]
    metadata = [{"page": page_num} for page_num, _ in chunks]

    print("📌 Genererer embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"✅ FAISS index laget med {len(embeddings)} vektorer.")
    return index, metadata

if __name__ == "__main__":
    pdf_file = "UD-21.pdf"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_file = "faiss_index.idx"
    metadata_file = "faiss_metadata.pkl"
    chunks_file = "chunks.txt"

    # 1) Del PDF
    chunks = pdf_to_chunks_per_page(pdf_file)

    # 2) Last inn modell
    model = SentenceTransformer(embedding_model_name)

    # 3) Lag index
    index, metadata = create_faiss_index(chunks, model)

    # 4) Lagre index + metadata
    faiss.write_index(index, faiss_index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    # 5) Lagre chunks.txt for Flask
    with open(chunks_file, "w", encoding="utf-8") as f:
        for page_num, text in chunks:
            safe_text = text.replace("\n", " ").replace("|", " ")
            f.write(f"{page_num}||{safe_text}\n")

    print(f"✅ Lagret index, metadata og {chunks_file}.")
