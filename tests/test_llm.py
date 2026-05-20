# Internal libs
from config import LLM_API_KEY, LLM_MODEL
from llm.gemini import GeminiClient
from schema import Query, Context, Response


def test_llm_config() -> None:
    assert LLM_API_KEY, "LLM_API_KEY is not set."
    assert LLM_MODEL, "LLM_MODEL is not set."


def test_llm_reachability() -> None:
    client = GeminiClient(api_key=LLM_API_KEY, model=LLM_MODEL)
    context = Context(Query("Ping", []), [])
    response = client.generate(context)

    assert isinstance(response, Response)
    assert response.text and response.text.strip()
    print(f"\nUser: {context.query.text}\nLLM: {response.text}")
