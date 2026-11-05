from langchain_core.language_models.chat_models import BaseChatModel


def make_chat_model(model: str, base_url: str, api_key: str, temperature: float = 0.2) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    # The OpenAI client treats an empty key as missing, even for endpoints that never check it.
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key or "not-needed", temperature=temperature)
