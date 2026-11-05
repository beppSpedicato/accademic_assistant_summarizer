import os
from pathlib import Path
from typing import List

import pytest
from chromadb.api.client import SharedSystemClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from academic_assistant import config as config_mod
from academic_assistant import llm
from academic_assistant.config import Config, save_config
from academic_assistant.utils import chroma_utils

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Keep every test away from the real config, network embeddings and VS Code."""
    cfg_path = tmp_path / "user-config" / "config.toml"
    monkeypatch.setattr(config_mod, "config_path", lambda: cfg_path)
    monkeypatch.setattr(chroma_utils, "get_embedding_function", lambda: DeterministicFakeEmbedding(size=64))
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    yield
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def config_file(tmp_path) -> Path:
    """A saved per-machine config, as `config init` would leave it."""
    return save_config(
        Config(base_url="http://llm.test/v1", model="test-model", api_key="sk-test-0123456789", subject="numerical methods"),
        config_mod.config_path(),
    )


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    monkeypatch.chdir(path)
    return path


class FakeLLM:
    """Stands in for llm.make_chat_model; records connection args and prompts.

    The code builds one chat model per summary/question, so each connection
    answers with the next queued response.
    """

    def __init__(self) -> None:
        self.responses: List[str] = []
        self.connections: List[tuple] = []
        self.prompts: List[str] = []
        self.error: Exception = None

    def __call__(self, model, base_url, api_key, temperature=0.2):
        self.connections.append((model, base_url, api_key))
        fake = self

        class Recording(FakeListChatModel):
            def invoke(self, messages, *args, **kwargs):
                fake.prompts.append("\n".join(m.content for m in messages))
                if fake.error:
                    raise fake.error
                return super().invoke(messages, *args, **kwargs)

        return Recording(responses=[self.responses.pop(0) if self.responses else "# Fake answer"])


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(llm, "make_chat_model", fake)
    return fake
