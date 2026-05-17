# Standard libs
from io import BytesIO

# 3rdparty libs
from google.genai import Client
from google.genai.types import Part, Content, GenerateContentConfig as GenaiConfig
from PIL.Image import Image

# Internal libs
from config import LLM_API_KEY, LLM_MODEL


def _to_part(image: Image) -> Part:
    buffer: BytesIO = BytesIO()
    image.save(buffer, format="JPEG")
    return Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


class GeminiClient:
    def __init__(
        self,
        model: str = LLM_MODEL,
        system_instruction: str | None = None,
    ) -> None:
        self.model: str = model
        self.client: Client = Client(api_key=LLM_API_KEY)
        self.system_instruction: str | None = system_instruction

    def generate(self, query: str, images: list[Image] | None = None) -> str | None:
        parts: list[Part] = [
            *(_to_part(image) for image in (images or [])),
            Part.from_text(text=query),
        ]
        content: Content = Content(parts=parts, role="user")
        config: GenaiConfig = GenaiConfig(system_instruction=self.system_instruction)
        return self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=config,
        ).text
