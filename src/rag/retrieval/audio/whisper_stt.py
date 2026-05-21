# Standard libs
from pathlib import Path

# 3rdparty libs
import whisper
from numpy import ndarray
from torch import Tensor


class WhisperTranscriber:
    def __init__(self, model_name: str = "turbo") -> None:
        self.model = whisper.load_model(name=model_name)

    def get_and_trim_audio(self, audio_file: Path) -> ndarray:
        audio = whisper.load_audio(str(audio_file))
        return whisper.pad_or_trim(audio)

    def compute_mel_spectrogram(self, audio: ndarray) -> Tensor:
        return whisper.log_mel_spectrogram(
            audio=audio,
            n_mels=self.model.dims.n_mels,
        ).to(self.model.device)

    def get_probabilities(self, mel_spectrogram: Tensor) -> dict[str, float]:
        _, probabilities = self.model.detect_language(mel_spectrogram)
        return probabilities

    def detect_language(self, audio_file: Path) -> str:
        audio = self.get_and_trim_audio(audio_file)
        mel = self.compute_mel_spectrogram(audio)
        probabilities = self.get_probabilities(mel)
        return max(probabilities, key=probabilities.get)

    def transcribe(self, audio_file: Path) -> str:
        return self.model.transcribe(str(audio_file))["text"]
