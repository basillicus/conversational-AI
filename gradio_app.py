"""gradio_app.py

A lightweight Gradio UI that calls the FastAPI /chat endpoint.
Run uvicorn main:app --reload
Then in another terminal run: python gradio_app.py

This file demonstrates a minimal UI suitable for demoing.
"""

import gradio as gr
import requests
import json

API_URL = "http://127.0.0.1:8000/chat"


def chat_ui(prompt, mode, ab_prob, llm_model):
    """Call the FastAPI endpoint and return a formatted response for Gradio."""
    payload = {
        "prompt": prompt,
        "mode": mode,
        "ab_prob": float(ab_prob),
        "llm_model": llm_model,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        # print(data)
        # pretty printable output
        meta = f"Model: {data.get('chosen_model')} | Latency: {data.get('latency_ms'):.1f} ms | Intent: {data.get('intent')} | Conf: {data.get('confidence'):.2f} | Anomaly: {data.get('anomaly')}"
        return data.get("answer"), meta
    except Exception as e:
        return f"Error contacting API: {e}", ""


with gr.Blocks() as demo:
    gr.Markdown(
        "# AI Chat Lifecycle Demo\nUse this UI to interact with the chat endpoint.\nRun the FastAPI app first (uvicorn main:app --reload)."
    )

    with gr.Row():
        inp = gr.Textbox(lines=2, placeholder="Enter a chat prompt...")
        out = gr.Textbox(label="Answer", lines=4)
    with gr.Row():
        mode = gr.Dropdown(choices=["auto", "local", "llm"], value="auto", label="Mode")
        ab = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.0, label="A/B probability of using LLM"
        )
        lm = gr.Textbox(value="mistral", label="LLM model name (ollama)")
    send = gr.Button("Send")
    meta = gr.Textbox(label="Metadata")

    def on_send(prompt, mode, ab, lm):
        ans, m = chat_ui(prompt, mode, ab, lm)
        return ans, m

    send.click(on_send, inputs=[inp, mode, ab, lm], outputs=[out, meta])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
