# Standard libs
from io import BytesIO

# 3rdparty libs
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from PIL.Image import Image

# Internal libs
from schema import Audio, Context, Response


def _image_to_part(image: Image) -> Part:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def _audio_to_part(audio: Audio) -> Part:
    with open(audio.path) as audio_file:
        return Part.from_bytes(data=audio_file.read(), mime_type="audio/mp3")


def _make_parts(context: Context) -> list[Part]:
    return [
        *(_image_to_part(img) for img in (context.query.images or [])),
        *(_audio_to_part(aud) for aud in (context.query.audio or [])),
        *(_image_to_part(img) for img in (context.images)),
        Part.from_text(text=context.query.text),
    ]


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        system_instruction: str | None = None,
    ) -> None:
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.system_instruction = system_instruction

    def generate(self, context: Context) -> Response:
        parts = _make_parts(context)
        content = Content(parts=parts, role="user")
        config = GenerateContentConfig(system_instruction=self.system_instruction)
        text = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=config,
        ).text
        return Response(text=text)
