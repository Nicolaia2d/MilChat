# Lokal NorwAI Mistral 7B LLM Fullstack

Beskrivelse av hvordan systemet er satt opp:


- Hugging Face Transformers
- Mistral 7B-modell
- GPU-akselerasjon (CUDA)
- Gradio (interaktivt UI)
- Flask (REST API)
- FastAPI (REST API med Swagger)
- CLI-script
- Miljøsjekk (`collect_env.py`)

---

## **Hva er dette prosjektet?**

Jeg har laget en **lokal pipeline** som:

- Kjører **NorwAI Mistral 7B instruct** lokalt, uten skytjeneste.
- Bruker Hugging Face `transformers` til å laste og kjøre modellen.
- Kjører alt på din **lokale GPU** via `torch` (PyTorch) med `torch_dtype=torch.float16`.
- Bruker `accelerate` + `device_map="auto"` for å fordele vektene smart.
- Gir deg **CLI**, **web-UI** og **REST API** fra samme kilde.

---

## **Hva inneholder prosjektet?**

| Fil                | Beskrivelse                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| `main.py`          | Kjør et enkelt prompt fra terminal, lagrer svaret i `output.txt`.           |
| `gradio_app.py`    | Starter en nettleser-app på `http://127.0.0.1:7860` med input-boks og svar. |
| `flask_app.py`     | Starter en REST API på `http://127.0.0.1:5000` med POST `/generate`.        |
| `fastapi_app.py`   | Starter en REST API med Swagger på `http://127.0.0.1:8000/docs`.            |
| `test_flask.py`    | Python-script som tester Flask `/generate` automatisk.                      |
| `test_fastapi.py`  | Python-script som tester FastAPI `/generate` automatisk.                    |
| `collect_env.py`   | Skript som viser PyTorch, CUDA, GPU og VRAM-status.                         |
| `requirements.txt` | Alle avhengigheter samlet for pip install.                                  |
| `READMEDETAILS.md` | Denne dokumentasjonen.                                                      |

---

## **Token — Hugging Face**

- Du laget en **fine-grained access token** på [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- Du logget inn:
  ```bash
  huggingface-cli login
  ```

