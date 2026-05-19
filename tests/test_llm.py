# Internal libs
from config import LLM_MODEL
from llm.gemini import GeminiClient
from schema import LLMResponse


def test_llm_reachability() -> None:
    client = GeminiClient(model=LLM_MODEL)
    response = client.generate(query="Say hello")

    assert isinstance(response, LLMResponse)
    assert response.text and response.text.strip()
