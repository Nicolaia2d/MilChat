import fitz  # PyMuPDF

def pdf_to_chunks_per_page(pdf_path):
    """
    Les PDF og del innholdet i én chunk per side.
    Returnerer en liste med tuples: (side_nummer, tekst).
    """
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  # Hopp over tomme sider
            chunks.append((page_num, text))

    print(f"✅ PDF-en ble splittet i {len(chunks)} side-chunks.")
    return chunks

if __name__ == "__main__":
    pdf_file = "UD-21.pdf"
    chunks = pdf_to_chunks_per_page(pdf_file)

    for page_num, chunk in chunks[:3]:  # Skriv ut de 3 første for sjekk
        print(f"\n--- Side {page_num} ---\n{chunk[:300]}...\n")  # Begrens output for lesbarhet
