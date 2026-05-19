# Standard libs
from io import BytesIO

# 3rdparty libs
from google.genai import Client
from google.genai.types import Content, GenerateContentConfig, Part
from PIL.Image import Image

# Internal libs
from config import LLM_API_KEY, LLM_MODEL
from schema import LLMResponse


def _to_part(image: Image) -> Part:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


class GeminiClient:
    def __init__(
        self,
        model: str = LLM_MODEL,
        system_instruction: str | None = None,
    ) -> None:
        self.model = model
        self.client = Client(api_key=LLM_API_KEY)
        self.system_instruction = system_instruction

    def generate(self, query: str, images: list[Image] | None = None) -> LLMResponse:
        parts = [
            *(_to_part(img) for img in (images or [])),
            Part.from_text(text=query),
        ]
        content = Content(parts=parts, role="user")
        config = GenerateContentConfig(system_instruction=self.system_instruction)
        text = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=config,
        ).text
        return LLMResponse(text=text)
