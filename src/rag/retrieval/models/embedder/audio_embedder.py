# Standard libs
from pathlib import Path

# Internal libs
from schema import Embedding


class AudioEmbedder:
    def __init__(self, enable_fusion: bool = False, ckpt: Path | None = None) -> None:
        import laion_clap as clap  # lazy import

        self.model = clap.CLAP_Module(enable_fusion=enable_fusion)
        self.model.load_ckpt(ckpt=str(ckpt) if ckpt else None)

    def embed(self, paths: list[str], use_tensor: bool = False) -> list[Embedding]:
        return self.model.get_audio_embedding_from_filelist(paths, use_tensor).tolist()
