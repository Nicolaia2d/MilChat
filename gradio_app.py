from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gradio as gr

# Device og modell
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

# Funksjon for Gradio
def generate_text(prompt):
    outputs = generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )
    return outputs[0]["generated_text"]

# Bygg Gradio-UI
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=5, placeholder="Skriv inn prompt her..."),
    outputs="text",
    title="NorwAI Mistral 7B",
    description="Kjører lokalt på GPU med Accelerate"
)

iface.launch()
