from langchain_openai import ChatOpenAI

from academic_assistant.llm import make_chat_model


def test_make_chat_model_passes_connection_settings():
    model = make_chat_model("my-model", "http://llm.test/v1", "sk-123")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "my-model"
    assert model.openai_api_base == "http://llm.test/v1"
    assert model.openai_api_key.get_secret_value() == "sk-123"


def test_empty_api_key_uses_placeholder(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = make_chat_model("m", "http://llm.test/v1", "")
    assert model.openai_api_key.get_secret_value() == "not-needed"
