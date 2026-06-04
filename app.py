# Standard libs
from pathlib import Path
from typing import Any

# 3rdparty libs
import gradio
from numpy import ndarray
from PIL import Image as PILImage

# Internal libs
from rag.base import BaseRAG
from schema import Query


def create_app(rag: BaseRAG, transcriber: Any = None) -> gradio.Blocks:
    def respond(text: str, image: ndarray | None, audio: str | None) -> str:
        if audio is not None and transcriber is not None:
            audio_text = transcriber.transcribe(Path(audio))
            text = f"{text}\n{audio_text}" if text else audio_text

        query_image = PILImage.fromarray(image) if image is not None else None
        query = Query(
            texts=[text] if text else None,
            images=[query_image] if query_image else None,
        )

        response = rag.generate(query)
        return response.content or ""

    with gradio.Blocks(title="ZiRAG", theme=gradio.themes.Glass()) as demo:
        gradio.Markdown("# ZiRAG -- Multimodal RAG for technical documentation")
        with gradio.Row():
            text_input = gradio.Textbox(label="Error description")
            image_input = gradio.Image(label="Display photo")
        audio_input = gradio.Audio(
            sources=["microphone", "upload"],
            label="Audio",
            type="filepath",
        )
        output = gradio.Textbox(label="Response")
        button = gradio.Button("Search")
        button.click(
            fn=respond,
            inputs=[text_input, image_input, audio_input],
            outputs=output,
        )

    return demo
