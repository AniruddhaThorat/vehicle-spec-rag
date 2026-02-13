import gradio as gr
from src.pipeline import run_pipeline

def ui_pipeline(pdf_file, query):

    if pdf_file is None:
        return "Please upload a PDF."

    result = run_pipeline(pdf_file.name, query)
    return result


demo = gr.Interface(
    fn=ui_pipeline,
    inputs=[
        gr.File(label="Upload Service Manual PDF"),
        gr.Textbox(label="Enter specification query")
    ],
    outputs=gr.Textbox(label="Output", lines=20),
    title="Vehicle Spec Extraction using RAG + Local Mistral",
    description="Upload a vehicle workshop manual and extract structured technical specifications."
)

demo.launch()