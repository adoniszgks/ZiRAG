# Internal libs
from schema import Embedding


class AudioEmbedder:
    def __init__(self, enable_fusion: bool = False) -> None:
        import laion_clap as clap  # lazy import

        self.model = clap.CLAP_Module(enable_fusion=enable_fusion)
        self.model.load_ckpt()

    def embed_text(self, texts: list[str]) -> list[Embedding] | None:
        if texts:
            return self.model.get_text_embedding(texts).tolist()
        return None

    def embed_audio(self, audios: list[str]) -> list[Embedding] | None:
        if audios:
            return self.model.get_audio_embedding_from_filelist(audios).tolist()
        return None
