# Standard libs
from io import BytesIO

# 3rdparty libs
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from PIL.Image import Image

# Internal libs
from schema import Audio, Context, Response

_AUDIO_MIME_TYPES = {".mp3": "audio/mp3", ".wav": "audio/wav", ".ogg": "audio/ogg"}


def _texts_to_part(text: str) -> Part:
    return Part.from_text(text=text)


def _image_to_part(image: Image) -> Part:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def _audio_to_part(audio: Audio) -> Part:
    mime_type = _AUDIO_MIME_TYPES.get(audio.path.suffix.lower(), "audio/wav")
    with open(audio.path, "rb") as audio_file:
        return Part.from_bytes(data=audio_file.read(), mime_type=mime_type)


def _make_parts(context: Context) -> list[Part]:
    return [
        *(_texts_to_part(txt) for txt in (context.query.texts or [])),
        *(_image_to_part(img) for img in (context.query.images or [])),
        *(_audio_to_part(aud) for aud in (context.query.audios or [])),
        *(_texts_to_part(txt) for txt in (context.texts or [])),
        *(_image_to_part(img) for img in (context.images or [])),
        *(_audio_to_part(aud) for aud in (context.audios or [])),
    ]


class GeminiLLM:
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
        if not parts:
            parts = [Part.from_text(text="[no input provided]")]
        content = Content(parts=parts, role="user")
        config = GenerateContentConfig(system_instruction=self.system_instruction)
        content = self.client.models.generate_content(
            model=self.model,
            contents=content,
            config=config,
        ).text or ""
        return Response(content=content)
