# 🧩 Hugging Face NorwAI Mistral 7B — Lokal kjøring

Dette prosjektet viser hvordan du:

- Kjører **NorwAI Mistral 7B** lokalt på GPU.
- Bruker **PyTorch** med CUDA.
- Laster modeller med `transformers` og `accelerate`.
- Har tre måter å kjøre modellen på: **CLI-script**, **Gradio-UI**, **Flask-API** og **FastAPI**.
- Sparer output til fil.

---

## 🚀 **Innhold**

| Fil              | Forklaring                                                        |
| ---------------- | ----------------------------------------------------------------- |
| `main.py`        | Kjører enkelt prompt fra terminal, lagrer output til `output.txt` |
| `gradio_app.py`  | Starter en **Gradio** nettleser-UI for interaktiv chatting        |
| `flask_app.py`   | Starter en **Flask REST API** på port 5000                        |
| `fastapi_app.py` | Starter en **FastAPI REST API** på port 8000 med Swagger UI       |
| `collect_env.py` | Verifiserer PyTorch, CUDA og GPU-status                           |

---

## ⚙️ **Krav**

**Installer alt med én kommando:**

```bash
pip install -r requirements.txt
```
