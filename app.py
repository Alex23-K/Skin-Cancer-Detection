import os, sys

# ───────────────────────────────────────────────────────────────────────────────
# Make sure the folder containing app.py is on Python's import path
cwd = os.path.dirname(os.path.abspath(__file__))
if cwd not in sys.path:
    sys.path.insert(0, cwd)
# ───────────────────────────────────────────────────────────────────────────────

import gradio as gr
from model import predict  # now guaranteed to find model.py next to app.py

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(source=["upload","webcam"], type="pil", label="Lesion Photo"),
        gr.Number(label="Age (optional)", optional=True),
        gr.Radio(["Male","Female"], label="Gender (optional)", optional=True)
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Confidence", precision=3)
    ],
    title="Skin Lesion Classifier",
    description="Upload or capture a photo. Age & gender are optional.",
    allow_flagging="never"
)

if __name__ == "__main__":
    # share=True will give you a public URL even if you're local, but on Hugging Face Spaces it's not needed.
    demo.launch()
