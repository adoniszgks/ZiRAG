# Standard libs
from pathlib import Path

# 3rdparty libs
import gradio as gr
import numpy as np
from PIL import Image as PILImage

# Internal libs
from rag.multimodal import MultimodalRAG
from schema import Audio, Query, Response

_LANGUAGES = ["English", "German", "French", "Spanish", "Italian", "Greek"]
_TITLE = "# Multimodal RAG for technical documentation"


class App:
    def __init__(self, mrag: MultimodalRAG) -> None:
        self.mrag = mrag

    def _format(self, response: Response) -> str:
        citations = "\n".join(
            f"{index} | "
            f"{citation.source:<5} | "
            f"{citation.filename} | "
            f"page {str('-' if citation.page is None else citation.page):>3} | "
            f"score {citation.score:.3f}"
            for index, citation in enumerate(response.citations)
        )
        if citations:
            return f"{response.content or ''}\n\nRetrieved sources:\n{citations}"
        return response.content or ""

    def _respond(
        self,
        text: str,
        image: np.ndarray | None,
        audio: str | None,
        use_textual: bool,
        use_visual: bool,
        use_aural: bool,
        language: str,
    ) -> str:
        query = Query(
            texts=[text] if text else None,
            images=[PILImage.fromarray(image)] if image is not None else None,
            audios=[Audio(path=Path(audio))] if audio else None,
        )
        response = self.mrag.generate(
            query=query,
            language=language,
            use_textual=use_textual,
            use_visual=use_visual,
            use_aural=use_aural,
        )
        return self._format(response)

    @property
    def css(self) -> str:
        return (Path(__file__).parent / "style.css").read_text()

    @property
    def theme(self) -> gr.themes.Base:
        return gr.themes.Glass()

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="ZiRAG") as demo:
            # Title row
            gr.Markdown(_TITLE)

            # RAG selection row
            with gr.Row():
                with gr.Column(scale=1):
                    use_textual = gr.Checkbox(label="Textual RAG", value=True)
                with gr.Column(scale=1):
                    use_visual = gr.Checkbox(label="Visual RAG", value=True)
                with gr.Column(scale=1):
                    use_aural = gr.Checkbox(label="Aural RAG", value=True)

            # Input row
            with gr.Row():
                with gr.Column(scale=1):
                    text = gr.Textbox(
                        label="Text query",
                        visible=True,
                        elem_id="text-input",
                    )
                with gr.Column(scale=1):
                    image = gr.Image(
                        label="Image query",
                        visible=True,
                        elem_id="image-input",
                    )
                with gr.Column(scale=1):
                    audio = gr.Audio(
                        sources=["microphone", "upload"],
                        label="Audio query",
                        type="filepath",
                        format="wav",
                        streaming=False,
                        visible=True,
                        elem_id="audio-input",
                    )

            # Response row
            language = gr.Dropdown(
                choices=_LANGUAGES,
                value="German",
                label="Response language",
            )
            output = gr.Textbox(label="Response", elem_id="response-box")
            button = gr.Button("Search", elem_id="search-btn", variant="primary")

            use_textual.change(lambda x: gr.update(visible=x), use_textual, text)
            use_visual.change(lambda x: gr.update(visible=x), use_visual, image)
            use_aural.change(lambda x: gr.update(visible=x), use_aural, audio)

            button.click(
                fn=self._respond,
                inputs=[
                    text,
                    image,
                    audio,
                    use_textual,
                    use_visual,
                    use_aural,
                    language,
                ],
                outputs=output,
            )

        return demo
