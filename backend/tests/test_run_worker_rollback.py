import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

from deerflow.runtime.runs.manager import RunManager
from deerflow.runtime.runs.schemas import RunStatus
from deerflow.runtime.runs.worker import (
    RunContext,
    _agent_factory_supports_app_config,
    _build_runtime_context,
    _ensure_interrupted_title,
    _extract_llm_error_fallback_message,
    _install_runtime_context,
    _rollback_to_pre_run_checkpoint,
    _try_extract_from_message,
    run_agent,
)


class FakeCheckpointer:
    def __init__(self, *, put_result):
        self.adelete_thread = AsyncMock()
        self.aput = AsyncMock(return_value=put_result)
        self.aput_writes = AsyncMock()


def _make_checkpoint(checkpoint_id: str, messages: list[str], version: int):
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    checkpoint["channel_values"] = {"messages": messages}
    checkpoint["channel_versions"] = {"messages": version}
    return checkpoint


def test_build_runtime_context_includes_app_config_when_present():
    app_config = object()

    context = _build_runtime_context("thread-1", "run-1", None, app_config)

    assert context["thread_id"] == "thread-1"
    assert context["run_id"] == "run-1"
    assert context["app_config"] is app_config


def test_install_runtime_context_preserves_existing_thread_id_and_threads_app_config():
    app_config = object()
    config = {"context": {"thread_id": "caller-thread"}}

    _install_runtime_context(
        config,
        {
            "thread_id": "record-thread",
            "run_id": "run-1",
            "app_config": app_config,
        },
    )

    assert config["context"]["thread_id"] == "caller-thread"
    assert config["context"]["run_id"] == "run-1"
    assert config["context"]["app_config"] is app_config


@pytest.mark.anyio
async def test_run_agent_threads_explicit_app_config_into_config_only_factory():
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    bridge = SimpleNamespace(
        publish=AsyncMock(),
        publish_end=AsyncMock(),
        cleanup=AsyncMock(),
    )
    app_config = object()
    captured: dict[str, object] = {}

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            captured["astream_context"] = config["context"]
            yield {"messages": []}

    def factory(*, config):
        captured["factory_context"] = config["context"]
        return DummyAgent()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, app_config=app_config),
        agent_factory=factory,
        graph_input={},
        config={},
    )
    await asyncio.sleep(0)

    assert captured["factory_context"]["app_config"] is app_config
    assert captured["astream_context"]["app_config"] is app_config
    fetched = await run_manager.get(record.run_id)
    assert fetched is not None
    assert fetched.status == RunStatus.success
    bridge.publish_end.assert_awaited_once_with(record.run_id)
    bridge.cleanup.assert_awaited_once_with(record.run_id, delay=60)


@pytest.mark.anyio
async def test_run_agent_marks_llm_error_fallback_as_error_status():
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    bridge = SimpleNamespace(
        publish=AsyncMock(),
        publish_end=AsyncMock(),
        cleanup=AsyncMock(),
    )

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            yield {
                "messages": [
                    AIMessage(
                        content="The configured LLM provider is temporarily unavailable after multiple retries.",
                        additional_kwargs={
                            "deerflow_error_fallback": True,
                            "error_type": "APIConnectionError",
                            "error_reason": "transient",
                            "error_detail": "Connection error.",
                        },
                    )
                ]
            }

    def factory(*, config):
        return DummyAgent()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None),
        agent_factory=factory,
        graph_input={},
        config={},
    )

    fetched = await run_manager.get(record.run_id)
    assert fetched is not None
    assert fetched.status == RunStatus.error
    assert fetched.error == "Connection error."
    bridge.publish_end.assert_awaited_once_with(record.run_id)


@pytest.mark.anyio
async def test_run_agent_defaults_root_run_name_from_assistant_id():
    run_manager = RunManager()
    record = await run_manager.create("thread-1", assistant_id="lead_agent")
    bridge = SimpleNamespace(
        publish=AsyncMock(),
        publish_end=AsyncMock(),
        cleanup=AsyncMock(),
    )
    captured: dict[str, object] = {}

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            captured["astream_run_name"] = config["run_name"]
            yield {"messages": []}

    def factory(*, config):
        captured["factory_run_name"] = config["run_name"]
        return DummyAgent()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None),
        agent_factory=factory,
        graph_input={},
        config={},
    )

    assert captured["factory_run_name"] == "lead_agent"
    assert captured["astream_run_name"] == "lead_agent"


@pytest.mark.anyio
async def test_run_agent_defaults_root_run_name_from_context_agent_name():
    run_manager = RunManager()
    record = await run_manager.create("thread-1", assistant_id="lead_agent")
    bridge = SimpleNamespace(
        publish=AsyncMock(),
        publish_end=AsyncMock(),
        cleanup=AsyncMock(),
    )
    captured: dict[str, object] = {}

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            captured["astream_run_name"] = config["run_name"]
            yield {"messages": []}

    def factory(*, config):
        captured["factory_run_name"] = config["run_name"]
        return DummyAgent()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None),
        agent_factory=factory,
        graph_input={},
        config={"context": {"agent_name": "finalis"}},
    )

    assert captured["factory_run_name"] == "finalis"
    assert captured["astream_run_name"] == "finalis"


@pytest.mark.anyio
async def test_run_agent_defaults_root_run_name_from_configurable_agent_name():
    run_manager = RunManager()
    record = await run_manager.create("thread-1", assistant_id="lead_agent")
    bridge = SimpleNamespace(
        publish=AsyncMock(),
        publish_end=AsyncMock(),
        cleanup=AsyncMock(),
    )
    captured: dict[str, object] = {}

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            captured["astream_run_name"] = config["run_name"]
            yield {"messages": []}

    def factory(*, config):
        captured["factory_run_name"] = config["run_name"]
        return DummyAgent()

    await run_agent(
        bridge,
        run_manager,
        record,
        ctx=RunContext(checkpointer=None),
        agent_factory=factory,
        graph_input={},
        config={"configurable": {"agent_name": "finalis"}},
    )

    assert captured["factory_run_name"] == "finalis"
    assert captured["astream_run_name"] == "finalis"


@pytest.mark.anyio
async def test_rollback_restores_snapshot_without_deleting_thread():
    checkpointer = FakeCheckpointer(put_result={"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "restored-1"}})

    await _rollback_to_pre_run_checkpoint(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
        pre_run_checkpoint_id="ckpt-1",
        pre_run_snapshot={
            "checkpoint_ns": "",
            "checkpoint": {
                "id": "ckpt-1",
                "channel_versions": {"messages": 3},
                "channel_values": {"messages": ["before"]},
            },
            "metadata": {"source": "input"},
            "pending_writes": [
                ("task-a", "messages", {"content": "first"}),
                ("task-a", "status", "done"),
                ("task-b", "events", {"type": "tool"}),
            ],
        },
        snapshot_capture_failed=False,
    )

    checkpointer.adelete_thread.assert_not_awaited()
    checkpointer.aput.assert_awaited_once()
    restore_config, restored_checkpoint, restored_metadata, new_versions = checkpointer.aput.await_args.args
    assert restore_config == {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    assert restored_checkpoint["id"] != "ckpt-1"
    assert "channel_versions" in restored_checkpoint
    assert "channel_values" in restored_checkpoint
    assert restored_checkpoint["channel_versions"] == {"messages": 3}
    assert restored_checkpoint["channel_values"] == {"messages": ["before"]}
    assert restored_metadata == {"source": "input"}
    assert new_versions == {"messages": 3}
    assert checkpointer.aput_writes.await_args_list == [
        call(
            {"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "restored-1"}},
            [("messages", {"content": "first"}), ("status", "done")],
            task_id="task-a",
        ),
        call(
            {"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "restored-1"}},
            [("events", {"type": "tool"})],
            task_id="task-b",
        ),
    ]


@pytest.mark.anyio
async def test_rollback_restored_checkpoint_becomes_latest_with_real_checkpointer():
    checkpointer = InMemorySaver()
    thread_config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    before_checkpoint = _make_checkpoint("0001", ["before"], 1)
    before_config = checkpointer.put(thread_config, before_checkpoint, {"step": 1}, {"messages": 1})
    after_checkpoint = _make_checkpoint("0002", ["after"], 2)
    after_config = checkpointer.put(before_config, after_checkpoint, {"step": 2}, {"messages": 2})
    checkpointer.put_writes(after_config, [("messages", "pending-after")], task_id="task-after")

    await _rollback_to_pre_run_checkpoint(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
        pre_run_checkpoint_id="0001",
        pre_run_snapshot={
            "checkpoint_ns": "",
            "checkpoint": before_checkpoint,
            "metadata": {"step": 1},
            "pending_writes": [("task-before", "messages", "pending-before")],
        },
        snapshot_capture_failed=False,
    )

    latest = checkpointer.get_tuple(thread_config)

    assert latest is not None
    assert latest.config["configurable"]["checkpoint_id"] != "0001"
    assert latest.config["configurable"]["checkpoint_id"] != "0002"
    assert latest.checkpoint["channel_values"] == {"messages": ["before"]}
    assert latest.pending_writes == [("task-before", "messages", "pending-before")]
    assert ("task-after", "messages", "pending-after") not in latest.pending_writes


@pytest.mark.anyio
async def test_rollback_deletes_thread_when_no_snapshot_exists():
    checkpointer = FakeCheckpointer(put_result=None)

    await _rollback_to_pre_run_checkpoint(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
        pre_run_checkpoint_id=None,
        pre_run_snapshot=None,
        snapshot_capture_failed=False,
    )

    checkpointer.adelete_thread.assert_awaited_once_with("thread-1")
    checkpointer.aput.assert_not_awaited()
    checkpointer.aput_writes.assert_not_awaited()


@pytest.mark.anyio
async def test_rollback_raises_when_restore_config_has_no_checkpoint_id():
    checkpointer = FakeCheckpointer(put_result={"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}})

    with pytest.raises(RuntimeError, match="did not return checkpoint_id"):
        await _rollback_to_pre_run_checkpoint(
            checkpointer=checkpointer,
            thread_id="thread-1",
            run_id="run-1",
            pre_run_checkpoint_id="ckpt-1",
            pre_run_snapshot={
                "checkpoint_ns": "",
                "checkpoint": {"id": "ckpt-1", "channel_versions": {}},
                "metadata": {},
                "pending_writes": [("task-a", "messages", "value")],
            },
            snapshot_capture_failed=False,
        )

    checkpointer.adelete_thread.assert_not_awaited()
    checkpointer.aput.assert_awaited_once()
    checkpointer.aput_writes.assert_not_awaited()


@pytest.mark.anyio
async def test_rollback_normalizes_none_checkpoint_ns_to_root_namespace():
    checkpointer = FakeCheckpointer(put_result={"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "restored-1"}})

    await _rollback_to_pre_run_checkpoint(
        checkpointer=checkpointer,
        thread_id="thread-1",
        run_id="run-1",
        pre_run_checkpoint_id="ckpt-1",
        pre_run_snapshot={
            "checkpoint_ns": None,
            "checkpoint": {"id": "ckpt-1", "channel_versions": {}},
            "metadata": {},
            "pending_writes": [],
        },
        snapshot_capture_failed=False,
    )

    checkpointer.aput.assert_awaited_once()
    restore_config, restored_checkpoint, restored_metadata, new_versions = checkpointer.aput.await_args.args
    assert restore_config == {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    assert restored_checkpoint["id"] != "ckpt-1"
    assert restored_checkpoint["channel_versions"] == {}
    assert restored_metadata == {}
    assert new_versions == {}


@pytest.mark.anyio
async def test_rollback_raises_on_malformed_pending_write_not_a_tuple():
    """pending_writes containing a non-3-tuple item should raise RuntimeError."""
    checkpointer = FakeCheckpointer(put_result={"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "restored-1"}})

    with pytest.raises(RuntimeError, match="rollback failed: pending_write is not a 3-tuple"):
        await _rollback_to_pre_run_checkpoint(
            checkpointer=checkpointer,
            thread_id="thread-1",
            run_id="run-1",
            pre_run_checkpoint_id="ckpt-1",
            pre_run_snapshot={
                "checkpoint_ns": "",
                "checkpoint": {"id": "ckpt-1", "channel_versions": {}},
                "metadata": {},
                "pending_writes": [
                    ("task-a", "messages", "valid"),  # valid
                    ["only", "two"],  # malformed: only 2 elements
                ],
            },
            snapshot_capture_failed=False,
        )

    # aput succeeded but aput_writes should not be called due to malformed data
    checkpointer.aput.assert_awaited_once()
    checkpointer.aput_writes.assert_not_awaited()


@pytest.mark.anyio
async def test_rollback_raises_on_malformed_pending_write_non_string_channel():
    """pending_writes containing a non-string channel should raise RuntimeError."""
    checkpointer = FakeCheckpointer(put_result={"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "restored-1"}})

    with pytest.raises(RuntimeError, match="rollback failed: pending_write has non-string channel"):
        await _rollback_to_pre_run_checkpoint(
            checkpointer=checkpointer,
            thread_id="thread-1",
            run_id="run-1",
            pre_run_checkpoint_id="ckpt-1",
            pre_run_snapshot={
                "checkpoint_ns": "",
                "checkpoint": {"id": "ckpt-1", "channel_versions": {}},
                "metadata": {},
                "pending_writes": [
                    ("task-a", 123, "value"),  # malformed: channel is not a string
                ],
            },
            snapshot_capture_failed=False,
        )

    checkpointer.aput.assert_awaited_once()
    checkpointer.aput_writes.assert_not_awaited()


@pytest.mark.anyio
async def test_rollback_propagates_aput_writes_failure():
    """If aput_writes fails, the exception should propagate (not be swallowed)."""
    checkpointer = FakeCheckpointer(put_result={"configurable": {"thread_id": "thread-1", "checkpoint_ns": "", "checkpoint_id": "restored-1"}})
    # Simulate aput_writes failure
    checkpointer.aput_writes.side_effect = RuntimeError("Database connection lost")

    with pytest.raises(RuntimeError, match="Database connection lost"):
        await _rollback_to_pre_run_checkpoint(
            checkpointer=checkpointer,
            thread_id="thread-1",
            run_id="run-1",
            pre_run_checkpoint_id="ckpt-1",
            pre_run_snapshot={
                "checkpoint_ns": "",
                "checkpoint": {"id": "ckpt-1", "channel_versions": {}},
                "metadata": {},
                "pending_writes": [
                    ("task-a", "messages", "value"),
                ],
            },
            snapshot_capture_failed=False,
        )

    # aput succeeded, aput_writes was called but failed
    checkpointer.aput.assert_awaited_once()
    checkpointer.aput_writes.assert_awaited_once()


def test_agent_factory_supports_app_config_detects_supported_signature():
    def factory(*, config, app_config=None):
        return (config, app_config)

    assert _agent_factory_supports_app_config(factory) is True


def test_build_runtime_context_defaults_to_thread_and_run_id():
    ctx = _build_runtime_context("thread-1", "run-1", None)
    assert ctx == {"thread_id": "thread-1", "run_id": "run-1"}


def test_build_runtime_context_merges_caller_context():
    """Regression for issue #2677: keys from ``config['context']`` (e.g. ``agent_name``)
    must be merged into the Runtime's context so that ``ToolRuntime.context`` — which
    is what ``setup_agent`` reads — can see them."""
    caller_context = {"agent_name": "my-agent", "is_bootstrap": True, "model_name": "gpt-4"}

    ctx = _build_runtime_context("thread-1", "run-1", caller_context)

    assert ctx["thread_id"] == "thread-1"
    assert ctx["run_id"] == "run-1"
    assert ctx["agent_name"] == "my-agent"
    assert ctx["is_bootstrap"] is True
    assert ctx["model_name"] == "gpt-4"


def test_build_runtime_context_caller_cannot_override_thread_id_or_run_id():
    """A malicious or buggy caller must not be able to overwrite the worker-assigned
    ``thread_id`` / ``run_id`` by stuffing them into ``config['context']``."""
    caller_context = {"thread_id": "spoofed", "run_id": "spoofed", "agent_name": "ok"}

    ctx = _build_runtime_context("real-thread", "real-run", caller_context)

    assert ctx["thread_id"] == "real-thread"
    assert ctx["run_id"] == "real-run"
    assert ctx["agent_name"] == "ok"


def test_build_runtime_context_ignores_non_dict_caller_context():
    ctx = _build_runtime_context("thread-1", "run-1", "not-a-dict")
    assert ctx == {"thread_id": "thread-1", "run_id": "run-1"}


def test_agent_factory_supports_app_config_returns_false_when_signature_lookup_fails(monkeypatch):
    class BrokenCallable:
        def __call__(self, **kwargs):
            return kwargs

    monkeypatch.setattr("deerflow.runtime.runs.worker.inspect.signature", lambda _obj: (_ for _ in ()).throw(ValueError("boom")))

    assert _agent_factory_supports_app_config(BrokenCallable()) is False


# ---------------------------------------------------------------------------
# _extract_llm_error_fallback_message coverage
# ---------------------------------------------------------------------------


def test_try_extract_from_message_finds_fallback_on_message_object():
    msg = AIMessage(
        content="fallback",
        additional_kwargs={
            "deerflow_error_fallback": True,
            "error_detail": "Connection error.",
            "error_reason": "transient",
        },
    )
    assert _try_extract_from_message(msg) == "Connection error."


def test_try_extract_from_message_finds_fallback_on_dict():
    msg = {
        "content": "fallback",
        "additional_kwargs": {
            "deerflow_error_fallback": True,
            "error_detail": "Quota exceeded.",
        },
    }
    assert _try_extract_from_message(msg) == "Quota exceeded."


def test_try_extract_from_message_returns_none_for_normal_message():
    msg = AIMessage(content="hello")
    assert _try_extract_from_message(msg) is None


def test_extract_llm_error_fallback_message_large_state_chunk_no_fallback():
    """Normal-size state dict without fallback markers must not raise and should return None."""
    large_state = {
        "messages": [
            AIMessage(content="Hello!"),
            {"role": "user", "content": "Hi there"},
        ],
        "foo": "x" * 10_000,
        "bar": {"nested": {"deep": {"data": list(range(1000))}}},
        "baz": [{"id": i, "payload": "y" * 1000} for i in range(500)],
    }
    assert _extract_llm_error_fallback_message(large_state) is None


def test_extract_llm_error_fallback_message_finds_fallback_in_messages_list():
    state = {
        "messages": [
            AIMessage(content="Hello!"),
            AIMessage(
                content="Unavailable.",
                additional_kwargs={
                    "deerflow_error_fallback": True,
                    "error_detail": "Connection error.",
                },
            ),
        ],
        "other_state": "large_value" * 1000,
    }
    assert _extract_llm_error_fallback_message(state) == "Connection error."


def test_extract_llm_error_fallback_message_finds_fallback_in_raw_message():
    msg = AIMessage(
        content="Unavailable.",
        additional_kwargs={
            "deerflow_error_fallback": True,
            "error_reason": "quota",
        },
    )
    assert _extract_llm_error_fallback_message(msg) == "quota"


def test_extract_llm_error_fallback_message_finds_fallback_in_tuple():
    item = (
        "messages",
        AIMessage(
            content="Unavailable.",
            additional_kwargs={
                "deerflow_error_fallback": True,
                "error_detail": "Circuit open.",
            },
        ),
    )
    assert _extract_llm_error_fallback_message(item) == "Circuit open."


def test_extract_llm_error_fallback_message_returns_none_for_empty_values():
    assert _extract_llm_error_fallback_message({}) is None
    assert _extract_llm_error_fallback_message([]) is None
    assert _extract_llm_error_fallback_message(None) is None
    assert _extract_llm_error_fallback_message("string") is None


def test_extract_llm_error_fallback_message_finds_fallback_in_updates_mode():
    """stream_mode='updates' yields dicts keyed by node name (e.g. {'call_model': {...}}).
    Fallback marker is nested inside the node's state update, not at the top level."""
    update_chunk = {
        "call_model": {
            "messages": [
                AIMessage(
                    content="Unavailable.",
                    additional_kwargs={
                        "deerflow_error_fallback": True,
                        "error_detail": "Connection error.",
                    },
                )
            ]
        }
    }
    assert _extract_llm_error_fallback_message(update_chunk) == "Connection error."


def test_extract_llm_error_fallback_message_updates_mode_no_fallback():
    """Normal updates chunk without any fallback should return None safely."""
    update_chunk = {
        "__interrupt__": [
            {
                "value": "ask_human",
                "resumable": True,
                "ns": ["agent"],
                "when": "during",
            }
        ]
    }
    assert _extract_llm_error_fallback_message(update_chunk) is None


class _FakeCheckpointTuple:
    """Minimal stand-in for ``CheckpointTuple`` used by ``_ensure_interrupted_title``."""

    def __init__(self, *, checkpoint: dict, metadata: dict, config: dict | None = None):
        self.checkpoint = checkpoint
        self.metadata = metadata
        self.config = config or {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}


class _TitleCheckpointer:
    """Captures ``aput`` arguments and exposes ``get_next_version`` like DB savers."""

    def __init__(self, *, tuple_value: _FakeCheckpointTuple | None, put_result: dict | None = None):
        self.aget_tuple = AsyncMock(return_value=tuple_value)
        self.aput = AsyncMock(return_value=put_result or {})

    def get_next_version(self, current, _channel):
        if current is None:
            return 1
        if isinstance(current, int):
            return current + 1
        if isinstance(current, str):
            try:
                return str(int(current) + 1)
            except ValueError:
                return f"{current}.1"
        return 1


@pytest.mark.anyio
async def test_ensure_interrupted_title_bumps_channel_version_and_declares_it_in_new_versions(monkeypatch):
    """Regression for #3859 review: DB-backed savers (Sqlite/Postgres) strip inline
    ``channel_values`` from ``put`` and only persist blobs for channels listed in
    ``new_versions``. The helper must therefore bump ``channel_versions["title"]``
    and pass ``{"title": next_version}`` so the fallback title actually survives
    a fresh ``aget_tuple`` after the worker's finally hook.
    """
    from deerflow.agents.middlewares.title_middleware import TitleMiddleware

    monkeypatch.setattr(
        TitleMiddleware,
        "_generate_title_result",
        lambda self, state, allow_partial_exchange=False: {"title": "Generated Title"},
    )

    initial_checkpoint = {
        "id": "ckpt-1",
        "ts": "2026-06-29T00:00:00Z",
        "channel_values": {"messages": [{"type": "human", "content": "hi"}]},
        "channel_versions": {"messages": 5},
    }
    checkpointer = _TitleCheckpointer(
        tuple_value=_FakeCheckpointTuple(
            checkpoint=initial_checkpoint,
            metadata={"source": "loop", "step": 7},
        ),
    )

    title = await _ensure_interrupted_title(checkpointer=checkpointer, thread_id="thread-1", app_config=None)

    assert title == "Generated Title"
    checkpointer.aput.assert_awaited_once()
    write_config, written_checkpoint, written_metadata, new_versions = checkpointer.aput.await_args.args

    # The title channel must be declared in new_versions — without this, DB
    # savers drop the inline channel_values["title"] from the persisted blob.
    assert new_versions == {"title": 1}
    # Channel versions on the checkpoint itself must also reflect the bump,
    # so a subsequent aget_tuple reconstructs channel_values with the title.
    assert written_checkpoint["channel_versions"]["title"] == 1
    # Pre-existing channel versions must be preserved.
    assert written_checkpoint["channel_versions"]["messages"] == 5
    # The fallback title rides into channel_values for the (legacy / single-table)
    # savers that inline the snapshot.
    assert written_checkpoint["channel_values"]["title"] == "Generated Title"
    assert written_metadata["source"] == "update"
    assert written_metadata["step"] == 8
    assert written_metadata["writes"] == {"runtime_interrupt_title": {"title": "Generated Title"}}
    assert write_config == {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}


@pytest.mark.anyio
async def test_ensure_interrupted_title_bumps_existing_string_version(monkeypatch):
    """When the checkpointer lacks ``get_next_version`` and the prior title
    version is a string (some savers use UUID-shaped versions), the helper must
    still produce a strictly different value rather than overwriting in place.
    """
    from deerflow.agents.middlewares.title_middleware import TitleMiddleware

    monkeypatch.setattr(
        TitleMiddleware,
        "_generate_title_result",
        lambda self, state, allow_partial_exchange=False: {"title": "T"},
    )

    initial_checkpoint = {
        "id": "ckpt-1",
        "ts": "2026-06-29T00:00:00Z",
        "channel_values": {"messages": [{"type": "human", "content": "hi"}]},
        "channel_versions": {"title": "v3"},
    }

    class _NoGetNextVersion:
        def __init__(self):
            self.aget_tuple = AsyncMock(
                return_value=_FakeCheckpointTuple(
                    checkpoint=initial_checkpoint,
                    metadata={},
                ),
            )
            self.aput = AsyncMock(return_value={})

    checkpointer = _NoGetNextVersion()
    await _ensure_interrupted_title(checkpointer=checkpointer, thread_id="thread-1", app_config=None)

    _, written_checkpoint, _, new_versions = checkpointer.aput.await_args.args
    bumped = written_checkpoint["channel_versions"]["title"]
    assert bumped != "v3", "title version must change so DB savers persist the update"
    assert new_versions == {"title": bumped}


@pytest.mark.anyio
async def test_ensure_interrupted_title_skips_when_title_already_set():
    """If the checkpoint already carries a title, no new checkpoint is written."""
    checkpointer = _TitleCheckpointer(
        tuple_value=_FakeCheckpointTuple(
            checkpoint={
                "id": "ckpt-1",
                "channel_values": {"messages": [], "title": "Already there"},
                "channel_versions": {"title": 1},
            },
            metadata={},
        ),
    )

    title = await _ensure_interrupted_title(checkpointer=checkpointer, thread_id="thread-1", app_config=None)

    assert title == "Already there"
    checkpointer.aput.assert_not_awaited()


@pytest.mark.anyio
async def test_ensure_interrupted_title_returns_none_when_no_checkpoint():
    """No checkpoint exists yet → nothing to update."""
    checkpointer = _TitleCheckpointer(tuple_value=None)
    assert await _ensure_interrupted_title(checkpointer=checkpointer, thread_id="thread-1", app_config=None) is None
    checkpointer.aput.assert_not_awaited()


@pytest.mark.anyio
async def test_ensure_interrupted_title_round_trip_with_real_sqlite_checkpointer(tmp_path):
    """Full round-trip against a real ``AsyncSqliteSaver`` on a disk-backed DB.

    Mirrors what Gateway constructs in production via ``make_checkpointer`` when
    ``database.backend == "sqlite"``, then closes and re-opens the saver to
    simulate a fresh connection. The fallback title must survive that boundary —
    this is the scenario the #3874 review flagged as broken before the
    ``new_versions={"title": ...}`` fix.
    """
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from deerflow.config.title_config import TitleConfig

    db_path = str(tmp_path / "ckpt.db")
    thread_cfg = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    # 1. Seed a first-turn checkpoint that has a human message and NO title —
    #    the same shape the agent leaves behind when interrupted mid-stream.
    async with AsyncSqliteSaver.from_conn_string(db_path) as writer:
        await writer.setup()
        ck = empty_checkpoint()
        ck["channel_values"] = {
            "messages": [HumanMessage(content="Why is the sky blue?").model_dump()],
        }
        ck["channel_versions"] = {"messages": 1}
        await writer.aput(thread_cfg, ck, {"source": "loop", "step": 1, "writes": {}}, {"messages": 1})

    # 2. Run the worker helper through a *fresh* saver instance — this is what
    #    the lifespan-owned checkpointer pool does for each request.
    title_config = TitleConfig(enabled=True, max_chars=40, max_words=20)
    app_config = SimpleNamespace(title=title_config)
    async with AsyncSqliteSaver.from_conn_string(db_path) as worker_saver:
        title = await _ensure_interrupted_title(
            checkpointer=worker_saver,
            thread_id="thread-1",
            app_config=app_config,
        )
    assert title, "fallback title must be generated from the seeded user message"

    # 3. Open ANOTHER fresh saver and confirm the title survives — this is the
    #    invariant the #3874 review was guarding: ``new_versions={}`` would
    #    cause DB savers to drop the title blob, so a fresh aget_tuple would
    #    read back without it.
    async with AsyncSqliteSaver.from_conn_string(db_path) as reader:
        tup = await reader.aget_tuple(thread_cfg)
    assert tup is not None
    persisted = tup.checkpoint.get("channel_values", {}).get("title")
    assert persisted == title
