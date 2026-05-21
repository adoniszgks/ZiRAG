# Standard libs
from pathlib import Path

# 3rdparty libs
import gradio
from numpy import ndarray
from PIL import Image as PILImage

# Internal libs
from pipeline import ZiRAG
from rag.retrieval.audio.whisper_stt import WhisperTranscriber
from schema import Query


def create_app(zirag: ZiRAG, transcriber: WhisperTranscriber) -> gradio.Blocks:
    def respond(text: str, image: ndarray | None, audio: str | None) -> str:

        if audio is not None:
            audio_text = transcriber.transcribe(Path(audio))
            text = f"{text}\n{audio_text}" if text else audio_text

        query_image = PILImage.fromarray(image) if image is not None else None
        query = Query(
            text=text or None,
            images=[query_image] if query_image else None,
        )

        response = zirag.generate(query)
        return response.text

    with gradio.Blocks(title="ZIRAG") as demo:
        gradio.Markdown("# ZIRAG -- Multimodal RAG for technical documentation")
        with gradio.Row():
            text_input = gradio.Textbox(label="Error description")
            image_input = gradio.Image(label="Display photo")
        audio_input = gradio.Audio(
            sources=["microphone", "upload"],
            label="Audio",
            type="filepath",
        )
        output = gradio.Textbox(label="Response")
        btn = gradio.Button("Search")
        btn.click(
            fn=respond,
            inputs=[text_input, image_input, audio_input],
            outputs=output,
        )

    return demo
