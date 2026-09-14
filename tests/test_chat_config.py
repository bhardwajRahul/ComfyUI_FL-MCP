import json

import pytest
from chat_config import PROVIDER_PRESETS, ChatSettingsStore, CredentialStore


def test_settings_store_contains_no_credentials(tmp_path):
    path = tmp_path / "settings.json"
    store = ChatSettingsStore(path)
    value = store.update({
        "provider": "lmstudio",
        "base_url": "http://127.0.0.1:1234/v1/",
        "model": "local-model",
    })
    assert value["base_url"] == "http://127.0.0.1:1234/v1"
    saved = json.loads(path.read_text())
    assert all("key" not in key.lower() for key in saved)


@pytest.mark.parametrize("provider, port", [("lmstudio", 1234), ("ollama", 11434)])
def test_local_openai_compatible_urls_add_v1_path(tmp_path, provider, port):
    store = ChatSettingsStore(tmp_path / "settings.json")

    value = store.update({
        "provider": provider,
        "base_url": f"http://192.168.1.10:{port}",
    })

    assert value["base_url"] == f"http://192.168.1.10:{port}/v1"


def test_custom_openai_compatible_url_preserves_root_path(tmp_path):
    store = ChatSettingsStore(tmp_path / "settings.json")

    value = store.update({
        "provider": "custom",
        "base_url": "http://192.168.1.10:8000",
    })

    assert value["base_url"] == "http://192.168.1.10:8000"


def test_settings_reject_secret_fields(tmp_path):
    store = ChatSettingsStore(tmp_path / "settings.json")
    with pytest.raises(ValueError, match="credential"):
        store.update({"api_key": "secret"})


def test_reasoning_effort_persists_and_validates(tmp_path):
    store = ChatSettingsStore(tmp_path / "settings.json")

    assert store.load()["reasoning_effort"] == "default"
    assert store.update({"reasoning_effort": "xhigh"})["reasoning_effort"] == "xhigh"
    assert store.load()["reasoning_effort"] == "xhigh"

    with pytest.raises(ValueError, match="reasoning effort"):
        store.update({"reasoning_effort": "unlimited"})


def test_search_mode_and_action_visibility_persist_and_validate(tmp_path):
    store = ChatSettingsStore(tmp_path / "settings.json")

    assert store.load()["search_mode"] == "free"
    value = store.update({
        "search_mode": "tavily_advanced",
        "show_action_buttons": False,
    })
    assert value["search_mode"] == "tavily_advanced"
    assert value["show_action_buttons"] is False

    with pytest.raises(ValueError, match="web search mode"):
        store.update({"search_mode": "surprise_me"})
    with pytest.raises(ValueError, match="true or false"):
        store.update({"show_action_buttons": "no"})


def test_tavily_credentials_use_the_same_secure_store(monkeypatch):
    store = CredentialStore()
    monkeypatch.setattr("keyring.set_password", lambda *_args: None)
    monkeypatch.setattr("keyring.get_password", lambda *_args: "tvly-secret")

    result = store.set("tavily", "tvly-secret")

    assert result["storage"] == "keychain"
    assert store.get("tavily") == "tvly-secret"
    with pytest.raises(ValueError, match="Unsupported provider"):
        store.set("unknown-search", "secret")


def test_approval_preferences_persist_and_validate(tmp_path):
    store = ChatSettingsStore(tmp_path / "settings.json")
    value = store.update({
        "approval_mode": "bypass_all",
        "always_allowed_tools": [
            "queue_workflow",
            "workflow_delete_file",
            "queue_workflow",
        ],
    })

    assert value["approval_mode"] == "bypass_all"
    assert value["always_allowed_tools"] == [
        "queue_workflow",
        "workflow_delete_file",
    ]
    assert store.load()["approval_mode"] == "bypass_all"

    with pytest.raises(ValueError, match="approval mode"):
        store.update({"approval_mode": "unsafe_unknown_mode"})
    with pytest.raises(ValueError, match="invalid tool name"):
        store.update({"always_allowed_tools": ["bad tool name"]})


def test_always_allow_tool_adds_one_persistent_rule(tmp_path):
    store = ChatSettingsStore(tmp_path / "settings.json")

    store.always_allow_tool("queue_workflow")
    store.always_allow_tool("queue_workflow")

    assert store.load()["always_allowed_tools"] == ["queue_workflow"]


def test_claude_subscription_is_separate_from_anthropic_api():
    subscription = PROVIDER_PRESETS["claude_subscription"]
    anthropic = PROVIDER_PRESETS["anthropic"]

    assert subscription["type"] == "claude_cli"
    assert subscription["requires_key"] is False
    assert subscription["default_model"] == "sonnet"
    assert [item["id"] for item in subscription["models"]] == [
        "default",
        "best",
        "fable",
        "sonnet",
        "opus",
        "haiku",
        "sonnet[1m]",
        "opus[1m]",
        "opusplan",
    ]
    assert anthropic["type"] == "anthropic"
    assert anthropic["requires_key"] is True


def test_claude_subscription_rejects_manually_stored_credentials():
    with pytest.raises(ValueError, match="managed by Claude Code"):
        CredentialStore().set("claude_subscription", "oauth-token")


def test_codex_subscription_is_separate_from_openai_api():
    subscription = PROVIDER_PRESETS["codex_subscription"]
    openai = PROVIDER_PRESETS["openai"]

    assert subscription["type"] == "codex_cli"
    assert subscription["requires_key"] is False
    assert subscription["default_model"] == "gpt-5.6-sol"
    assert [item["id"] for item in subscription["models"]] == [
        "gpt-5.6-sol",
        "codex-auto-review",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex-spark",
    ]
    assert openai["type"] == "openai_compatible"
    assert openai["requires_key"] is True


def test_codex_subscription_rejects_manually_stored_credentials():
    with pytest.raises(ValueError, match="managed by Codex"):
        CredentialStore().set("codex_subscription", "oauth-token")
