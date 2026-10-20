"""Offline tests for optional Langfuse tracing (app/core/tracing.py)."""

import sys
from types import SimpleNamespace

import pytest
from conftest import settings
from pydantic import SecretStr

from app.core import tracing


@pytest.fixture(autouse=True)
def _clean_tracing_state():
    """Each test starts from a pristine client/template cache."""
    tracing._reset_for_tests()
    yield
    tracing._reset_for_tests()


@pytest.fixture
def fused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings with both Langfuse credentials present."""
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(
        settings, "langfuse_secret_key", SecretStr("sk-lf-test")
    )
    monkeypatch.setattr(
        settings, "langfuse_base_url", "https://cloud.langfuse.com"
    )


class _FakeLangfuse:
    """Stand-in for the real client: records construction args and shutdown."""

    instances: list["_FakeLangfuse"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.shut_down = False
        _FakeLangfuse.instances.append(self)

    def shutdown(self) -> None:
        self.shut_down = True


class _FakeHandler:
    """Stand-in for `langfuse.langchain.CallbackHandler`."""

    def __init__(self, public_key: str | None = None, **_kwargs: object) -> None:
        self.public_key = public_key


def _stub_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fakes at the import locations tracing.py uses lazily."""
    module = SimpleNamespace(Langfuse=_FakeLangfuse)
    handler_mod = SimpleNamespace(CallbackHandler=_FakeHandler)
    monkeypatch.setitem(sys.modules, "langfuse", module)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", handler_mod)


class TestLangfuseEnabled:
    def test_disabled_without_credentials(self) -> None:
        assert tracing.langfuse_enabled(settings) is False

    def test_enabled_with_both_keys(self, fused: None) -> None:
        assert tracing.langfuse_enabled(settings) is True

    def test_disabled_with_only_public_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
        assert tracing.langfuse_enabled(settings) is False

    def test_disabled_with_only_secret_key(self, monkeypatch) -> None:
        monkeypatch.setattr(
            settings, "langfuse_secret_key", SecretStr("sk-lf-test")
        )
        assert tracing.langfuse_enabled(settings) is False


class TestClientCredentials:
    """Credential invariants must be typed errors, not `assert` (python -O safe)."""

    def test_missing_public_key_raises_typed_error(self, fused: None, monkeypatch) -> None:
        monkeypatch.setattr(settings, "langfuse_public_key", None)
        with pytest.raises(RuntimeError):
            tracing._get_or_create_client(settings)

    def test_missing_secret_key_raises_typed_error(self, fused: None, monkeypatch) -> None:
        monkeypatch.setattr(settings, "langfuse_secret_key", None)
        with pytest.raises(RuntimeError):
            tracing._get_or_create_client(settings)


class TestBuildTraceCallbacks:
    def test_returns_empty_when_disabled(self) -> None:
        assert tracing.build_trace_callbacks(settings) == []

    def test_attaches_handler_and_caches_client(
        self, fused: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_langfuse(monkeypatch)
        _FakeLangfuse.instances.clear()

        callbacks = tracing.build_trace_callbacks(settings)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], _FakeHandler)
        assert callbacks[0].public_key == "pk-lf-test"

        # The client is built once with the mask and the configured base URL.
        assert len(_FakeLangfuse.instances) == 1
        client = _FakeLangfuse.instances[0]
        assert client.kwargs["public_key"] == "pk-lf-test"
        assert client.kwargs["secret_key"] == "sk-lf-test"
        assert client.kwargs["base_url"] == "https://cloud.langfuse.com"
        assert callable(client.kwargs["mask"])

        # Second request reuses the cached client, gets a fresh handler.
        again = tracing.build_trace_callbacks(settings)
        assert len(_FakeLangfuse.instances) == 1
        assert again[0] is not callbacks[0]

    def test_import_failure_degrades_to_no_tracing(
        self, fused: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "langfuse", None)
        assert tracing.build_trace_callbacks(settings) == []

    def test_client_init_failure_degrades_to_no_tracing(
        self, fused: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(**_kwargs: object) -> None:
            raise RuntimeError("unreachable langfuse backend")

        _stub_langfuse(monkeypatch)
        monkeypatch.setitem(
            sys.modules,
            "langfuse",
            SimpleNamespace(Langfuse=_boom),
        )
        assert tracing.build_trace_callbacks(settings) == []


class TestConfigureAndShutdown:
    def test_noop_when_disabled(self, monkeypatch) -> None:
        _stub_langfuse(monkeypatch)
        _FakeLangfuse.instances.clear()
        tracing.configure_langfuse_environment(settings)
        assert _FakeLangfuse.instances == []

    def test_init_failure_is_tolerated(
        self, fused: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            sys.modules, "langfuse", SimpleNamespace(Langfuse=lambda **_k: 1 / 0)
        )
        tracing.configure_langfuse_environment(settings)  # must not raise

    def test_shutdown_flushes_all_clients(
        self, fused: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_langfuse(monkeypatch)
        tracing.build_trace_callbacks(settings)
        client = _FakeLangfuse.instances[0]
        tracing.shutdown_tracing()
        assert client.shut_down is True
        assert tracing._clients == {}

    def test_shutdown_tolerates_client_errors(
        self, fused: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_langfuse(monkeypatch)
        tracing.build_trace_callbacks(settings)

        def _boom() -> None:
            raise RuntimeError("flush failed")

        monkeypatch.setattr(_FakeLangfuse.instances[0], "shutdown", _boom)
        tracing.shutdown_tracing()  # must not raise


class TestMask:
    def test_replaces_registered_template_everywhere(self) -> None:
        tracing.register_prompt_template("planner", "v1", "SECRET TEMPLATE")
        data = {
            "messages": [{"role": "system", "content": "SECRET TEMPLATE"}],
            "nested": [["prefix SECRET TEMPLATE suffix"]],
            "untouched": 42,
        }
        masked = tracing._mask(data=data)
        rendered = str(masked)
        assert "SECRET TEMPLATE" not in rendered
        assert rendered.count("[prompt:planner@v1") == 2
        assert masked["untouched"] == 42

    def test_passthrough_without_registrations(self) -> None:
        data = {"content": "anything"}
        assert tracing._mask(data=data) == data

    def test_longest_template_wins_over_its_prefix(self) -> None:
        """A v2 extending v1's text must mask whole, not leave a suffix behind."""
        tracing.register_prompt_template("planner", "v1", "BASE TEMPLATE")
        tracing.register_prompt_template(
            "planner", "v2", "BASE TEMPLATE\nExtra tail."
        )
        masked = tracing._mask(data={"content": "BASE TEMPLATE\nExtra tail."})
        rendered = str(masked)
        assert "BASE TEMPLATE" not in rendered
        assert "Extra tail" not in rendered
        assert "[prompt:planner@v2" in rendered

    def test_tuples_and_plain_values(self) -> None:
        tracing.register_prompt_template("planner", "v2", "T")
        placeholder = tracing._template_placeholders["T"]
        assert placeholder.startswith("[prompt:planner@v2 sha256:")
        masked = tracing._mask(data=("T", None, 3.5, "other"))
        assert masked == [placeholder, None, 3.5, "other"]


class TestRunConfig:
    def test_callbacks_empty_offline(self, client) -> None:
        """Conftest clears keys, so _run_config attaches no handler."""
        from app.api.routes import _run_config

        request = SimpleNamespace(
            state=SimpleNamespace(request_id="r1"),
            app=SimpleNamespace(
                state=SimpleNamespace(
                    prompt_registry=client.app.state.prompt_registry
                )
            ),
        )
        config = _run_config(request, "sess-1", ["agentic-api"])
        assert config["callbacks"] == []
        assert config["metadata"]["langfuse_session_id"] == "sess-1"
        assert config["metadata"]["langfuse_tags"] == [
            "agentic-api",
            "prompt:planner@v1",
        ]
        assert config["metadata"]["prompt_version"] == "v1"

    def test_secret_never_in_settings_repr(self) -> None:
        assert "sk-lf" not in repr(settings)
