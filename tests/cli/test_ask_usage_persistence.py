"""ask-mode usage persistence: with --session the journal gains the accounting
marker and internal llm.* events; without persistence nothing attaches."""

from kolega_code.cli.session_usage import LLM_MESSAGE_EVENT, LLM_RUN_STARTED_EVENT
from kolega_code.llm.ledger import LlmCallOrigin, llm_call_origin
from kolega_code.llm.models import Message
from kolega_code.llm.usage import normalize_usage

from ._app_test_utils import FakeCoderAgent


class UsageRecordingAskAgent(FakeCoderAgent):
    """Simulates a sub-agent settlement on the shared ledger during the turn."""

    agent_name = "coder"
    instances: list["UsageRecordingAskAgent"] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        UsageRecordingAskAgent.instances.append(self)

    async def process_message_stream(self, message, attachments=None):
        ledger = self.kwargs["usage_ledger"]
        usage = normalize_usage({"input_tokens": 10, "output_tokens": 5}, "anthropic", "m")
        with llm_call_origin(LlmCallOrigin(kind="sub_agent", agent_name="Investigator", agent_id="a1")):
            request_id = ledger.begin("anthropic", "m")
        ledger.record_response(request_id, usage, message=Message(role="assistant", content="sub", usage=usage))
        async for item in FakeCoderAgent.process_message_stream(self, message, attachments):
            yield item

    async def fire_hook(self, event, payload):
        class Result:
            additional_context = None
            blocked = False
            end_turn = False

        return Result()


def _setup(tmp_path, monkeypatch):
    from kolega_code.cli import main as main_module

    UsageRecordingAskAgent.instances = []
    monkeypatch.setattr(main_module, "CoderAgent", UsageRecordingAskAgent)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KOLEGA_CODE_PROVIDER", "anthropic")
    return main_module, project


def test_ask_with_session_journals_marker_and_internal_events(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project), "--session", "s1"])
    assert exit_code == 0

    from kolega_code.cli.session_store import SessionStore, default_state_dir

    store = SessionStore(default_state_dir())
    sessions = store.list()
    assert len(sessions) == 1
    events = store.journal(sessions[0].session_id).read_events()
    types = [e.event_type for e in events]
    assert types.count(LLM_RUN_STARTED_EVENT) == 1
    # The marker precedes the turn it covers.
    assert types.index(LLM_RUN_STARTED_EVENT) < types.index("turn.started")
    llm_messages = [e for e in events if e.event_type == LLM_MESSAGE_EVENT]
    assert len(llm_messages) == 1
    assert llm_messages[0].payload["origin"]["agent_name"] == "Investigator"

    record = store.load(sessions[0].session_id)
    assert record.usage["responses"] >= 1
    assert record.usage["coverage"]["full"] is True


def test_ask_without_persistence_attaches_no_observer(tmp_path, monkeypatch, isolated_cli_env):
    main_module, project = _setup(tmp_path, monkeypatch)

    exit_code = main_module.main(["ask", "do the thing", "--project", str(project)])
    assert exit_code == 0
    agent = UsageRecordingAskAgent.instances[0]
    assert agent.kwargs["usage_ledger"].observer is None
