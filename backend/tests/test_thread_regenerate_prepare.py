from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import empty_checkpoint, uuid6
from langgraph.checkpoint.memory import InMemorySaver

from deerflow.runtime import RunStatus
from deerflow.utils.messages import ORIGINAL_USER_CONTENT_KEY


def _checkpoint(checkpoint_id: str, messages: list[object], *, metadata: dict | None = None):
    return SimpleNamespace(
        config={
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
                "checkpoint_map": None,
            }
        },
        checkpoint={"channel_values": {"messages": messages}},
        metadata=metadata or {},
    )


async def _put_memory_checkpoint(
    checkpointer: InMemorySaver,
    thread_id: str,
    messages: list[object],
    *,
    step: int,
    parent_config: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = str(uuid6())
    checkpoint["channel_values"] = {"messages": messages}
    checkpoint["channel_versions"] = {"messages": step}
    checkpoint_metadata = {
        "step": step,
        "source": "loop",
        "writes": {"test": {"messages": messages}},
        "parents": {},
    }
    checkpoint_metadata.update(metadata or {})
    return await checkpointer.aput(
        parent_config or {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        checkpoint,
        checkpoint_metadata,
        {"messages": step},
    )


async def _collect_checkpoints(checkpointer: InMemorySaver, config: dict) -> list:
    return [checkpoint async for checkpoint in checkpointer.alist(config)]


class FakeCheckpointer:
    def __init__(self, history, *, latest=None):
        self.history = history
        self.latest = latest
        self.alist_limits = []

    async def aget_tuple(self, config):
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        if checkpoint_id:
            return next((item for item in self.history if item.config["configurable"]["checkpoint_id"] == checkpoint_id), None)
        return self.latest or (self.history[0] if self.history else None)

    async def alist(self, config, limit=200):
        self.alist_limits.append(limit)
        for item in self.history[:limit]:
            yield item


class FakeEventStore:
    def __init__(self, rows):
        self.rows = rows

    async def list_messages(self, thread_id, *, limit=50, before_seq=None, after_seq=None):
        return self.rows[-limit:]


class ThreadAwareEventStore:
    def __init__(self, rows_by_thread):
        self.rows_by_thread = rows_by_thread
        self.calls = []

    async def list_messages(self, thread_id, *, limit=50, before_seq=None, after_seq=None):
        self.calls.append(thread_id)
        return self.rows_by_thread.get(thread_id, [])[-limit:]


class FakeRunManager:
    def __init__(self, records):
        self.records = records

    async def list_by_thread(self, thread_id, *, user_id=None, limit=100):
        return self.records[:limit]


class FakeThreadStore:
    def __init__(self, inaccessible: set[str] | None = None):
        self.inaccessible = inaccessible or set()

    async def get(self, thread_id):
        return None if thread_id in self.inaccessible else {"thread_id": thread_id}


def _request(checkpointer, event_store, *, run_manager=None, thread_store=None, user_id="user-1"):
    from app.gateway.auth_disabled import AUTH_SOURCE_SESSION

    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                checkpointer=checkpointer,
                run_event_store=event_store,
                run_manager=run_manager or FakeRunManager([]),
                thread_store=thread_store or FakeThreadStore(),
            )
        ),
        state=SimpleNamespace(user=SimpleNamespace(id=user_id), auth_source=AUTH_SOURCE_SESSION),
    )


def test_prepare_regenerate_payload_returns_clean_input_and_base_checkpoint():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(
        id="human-1",
        content="<uploaded_files>injected</uploaded_files>\n\n/data-analysis analyze data.csv",
        additional_kwargs={
            ORIGINAL_USER_CONTENT_KEY: "/data-analysis analyze data.csv",
            "files": [{"filename": "data.csv", "path": "/mnt/user-data/uploads/data.csv"}],
        },
    )
    ai = AIMessage(id="ai-1", content="answer v1")
    base = _checkpoint("ckpt-base", [])
    after_human = _checkpoint("ckpt-human", [human])
    latest = _checkpoint("ckpt-ai", [human, ai])
    checkpointer = FakeCheckpointer([latest, after_human, base])
    event_store = FakeEventStore(
        [
            {
                "run_id": "run-old",
                "event_type": "llm.ai.response",
                "category": "message",
                "content": {"id": "ai-1", "type": "ai", "content": "answer v1"},
                "metadata": {"caller": "lead_agent"},
            }
        ]
    )

    response = asyncio.run(_prepare_regenerate_payload("thread-1", "ai-1", _request(checkpointer, event_store)))

    assert response.checkpoint == {
        "checkpoint_ns": "",
        "checkpoint_id": "ckpt-base",
        "checkpoint_map": None,
    }
    assert response.target_run_id == "run-old"
    assert response.metadata == {
        "regenerate_from_message_id": "ai-1",
        "regenerate_from_run_id": "run-old",
        "regenerate_checkpoint_id": "ckpt-base",
    }
    regenerated_human = response.input["messages"][0]
    assert regenerated_human["id"] == "human-1"
    assert regenerated_human["content"] == [{"type": "text", "text": "/data-analysis analyze data.csv"}]
    assert regenerated_human["additional_kwargs"] == {"files": [{"filename": "data.csv", "path": "/mnt/user-data/uploads/data.csv"}]}


def test_prepare_regenerate_payload_repairs_legacy_single_checkpoint_branch_idempotently():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    checkpointer = InMemorySaver()
    source_thread_id = "source-thread"
    branch_thread_id = "legacy-branch"
    source_run_id = "source-run"
    human = HumanMessage(id="human-1", content="question", additional_kwargs={"run_id": source_run_id})
    ai = AIMessage(id="ai-1", content="answer")

    async def _seed() -> tuple[str, str]:
        source_base_config = await _put_memory_checkpoint(checkpointer, source_thread_id, [], step=0)
        after_human = await _put_memory_checkpoint(
            checkpointer,
            source_thread_id,
            [human],
            step=1,
            parent_config=source_base_config,
        )
        source_head_config = await _put_memory_checkpoint(
            checkpointer,
            source_thread_id,
            [human, ai],
            step=2,
            parent_config=after_human,
        )
        source_head = await checkpointer.aget_tuple(source_head_config)
        assert source_head is not None

        legacy_head = copy.deepcopy(source_head.checkpoint)
        legacy_head_id = str(uuid6())
        legacy_head["id"] = legacy_head_id
        legacy_metadata = copy.deepcopy(source_head.metadata)
        legacy_metadata.update(
            {
                "source": "branch",
                "deerflow_branch": True,
                "branch_parent_thread_id": source_thread_id,
                "branch_parent_checkpoint_id": source_head_config["configurable"]["checkpoint_id"],
                "branch_parent_message_id": "ai-1",
            }
        )
        await checkpointer.aput(
            {"configurable": {"thread_id": branch_thread_id, "checkpoint_ns": ""}},
            legacy_head,
            legacy_metadata,
            dict(legacy_head["channel_versions"]),
        )
        return source_base_config["configurable"]["checkpoint_id"], legacy_head_id

    source_base_id, legacy_head_id = asyncio.run(_seed())
    request = _request(checkpointer, FakeEventStore([]))

    first = asyncio.run(_prepare_regenerate_payload(branch_thread_id, "ai-1", request))
    second = asyncio.run(_prepare_regenerate_payload(branch_thread_id, "ai-1", request))

    assert first.checkpoint["checkpoint_id"] == source_base_id
    assert second.checkpoint == first.checkpoint
    assert first.target_run_id == source_run_id

    latest = asyncio.run(checkpointer.aget_tuple({"configurable": {"thread_id": branch_thread_id, "checkpoint_ns": ""}}))
    assert latest is not None
    assert latest.config["configurable"]["checkpoint_id"] == legacy_head_id
    branch_history = asyncio.run(_collect_checkpoints(checkpointer, {"configurable": {"thread_id": branch_thread_id, "checkpoint_ns": ""}}))
    assert [item.config["configurable"]["checkpoint_id"] for item in branch_history] == [legacy_head_id, source_base_id]


def test_prepare_regenerate_payload_rejects_legacy_branch_when_source_checkpoint_is_missing():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    checkpointer = InMemorySaver()
    branch_thread_id = "legacy-orphan"
    human = HumanMessage(id="human-1", content="question", additional_kwargs={"run_id": "source-run"})
    ai = AIMessage(id="ai-1", content="answer")

    async def _seed() -> None:
        await _put_memory_checkpoint(
            checkpointer,
            branch_thread_id,
            [human, ai],
            step=1,
            metadata={
                "source": "branch",
                "deerflow_branch": True,
                "branch_parent_thread_id": "deleted-source",
                "branch_parent_checkpoint_id": "missing-checkpoint",
                "branch_parent_message_id": "ai-1",
            },
        )

    asyncio.run(_seed())
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            _prepare_regenerate_payload(
                branch_thread_id,
                "ai-1",
                _request(checkpointer, FakeEventStore([])),
            )
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "Could not restore the regenerate checkpoint because the source branch checkpoint is unavailable"
    branch_history = asyncio.run(_collect_checkpoints(checkpointer, {"configurable": {"thread_id": branch_thread_id, "checkpoint_ns": ""}}))
    assert len(branch_history) == 1


def test_prepare_regenerate_payload_rejects_non_latest_assistant():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(id="human-1", content="question")
    old_ai = AIMessage(id="ai-old", content="old")
    latest_ai = AIMessage(id="ai-latest", content="latest")
    base = _checkpoint("ckpt-base", [])
    after_human = _checkpoint("ckpt-human", [human])
    latest = _checkpoint("ckpt-latest", [human, old_ai, latest_ai])
    checkpointer = FakeCheckpointer([latest, after_human, base])
    event_store = FakeEventStore(
        [
            {
                "run_id": "run-old",
                "event_type": "ai_message",
                "category": "message",
                "content": {"id": "ai-old", "type": "ai", "content": "old"},
                "metadata": {"caller": "lead_agent"},
            }
        ]
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_prepare_regenerate_payload("thread-1", "ai-old", _request(checkpointer, event_store)))

    assert exc.value.status_code == 409
    assert exc.value.detail == "Only the latest assistant message can be regenerated"


def test_prepare_regenerate_payload_falls_back_to_matching_run_when_events_are_missing():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(id="human-1", content="question")
    ai = AIMessage(id="ai-1", content="answer")
    base = _checkpoint("ckpt-base", [])
    after_human = _checkpoint("ckpt-human", [human])
    latest = _checkpoint("ckpt-ai", [human, ai])
    checkpointer = FakeCheckpointer([latest, after_human, base])
    run_manager = FakeRunManager(
        [
            SimpleNamespace(run_id="run-latest", status=RunStatus.success, last_ai_message="answer"),
            SimpleNamespace(run_id="run-older", status=RunStatus.error, last_ai_message="answer"),
        ]
    )

    response = asyncio.run(
        _prepare_regenerate_payload(
            "thread-1",
            "ai-1",
            _request(checkpointer, FakeEventStore([]), run_manager=run_manager),
        )
    )

    assert response.target_run_id == "run-latest"
    assert response.metadata["regenerate_from_run_id"] == "run-latest"


def test_prepare_regenerate_payload_falls_back_to_parent_branch_events():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(id="human-1", content="question")
    ai = AIMessage(id="ai-1", content="answer")
    base = _checkpoint("ckpt-base", [])
    after_human = _checkpoint("ckpt-human", [human])
    latest = _checkpoint(
        "ckpt-ai",
        [human, ai],
        metadata={
            "deerflow_branch": True,
            "branch_parent_thread_id": "parent-thread",
            "branch_parent_checkpoint_id": "parent-checkpoint",
        },
    )
    checkpointer = FakeCheckpointer([latest, after_human, base])
    event_store = ThreadAwareEventStore(
        {
            "parent-thread": [
                {
                    "run_id": "parent-run",
                    "event_type": "llm.ai.response",
                    "category": "message",
                    "content": {"id": "ai-1", "type": "ai", "content": "answer"},
                    "metadata": {"caller": "lead_agent"},
                }
            ]
        }
    )

    response = asyncio.run(_prepare_regenerate_payload("thread-1", "ai-1", _request(checkpointer, event_store)))

    assert response.target_run_id == "parent-run"


def test_prepare_regenerate_payload_does_not_read_inaccessible_parent_branch_events():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(id="human-1", content="question")
    ai = AIMessage(id="ai-1", content="answer")
    base = _checkpoint("ckpt-base", [])
    after_human = _checkpoint("ckpt-human", [human])
    latest = _checkpoint(
        "ckpt-ai",
        [human, ai],
        metadata={
            "deerflow_branch": True,
            "branch_parent_thread_id": "other-users-thread",
            "branch_parent_checkpoint_id": "parent-checkpoint",
        },
    )
    checkpointer = FakeCheckpointer([latest, after_human, base])
    event_store = ThreadAwareEventStore(
        {
            "other-users-thread": [
                {
                    "run_id": "private-parent-run",
                    "event_type": "llm.ai.response",
                    "category": "message",
                    "content": {"id": "ai-1", "type": "ai", "content": "answer"},
                }
            ]
        }
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            _prepare_regenerate_payload(
                "thread-1",
                "ai-1",
                _request(
                    checkpointer,
                    event_store,
                    thread_store=FakeThreadStore({"other-users-thread"}),
                ),
            )
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "Could not find source run for assistant message"
    assert event_store.calls == ["thread-1"]


def test_prepare_regenerate_payload_rejects_unverified_run_fallback_when_events_are_missing():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(id="human-1", content="question")
    ai = AIMessage(id="ai-1", content="answer")
    base = _checkpoint("ckpt-base", [])
    after_human = _checkpoint("ckpt-human", [human])
    latest = _checkpoint("ckpt-ai", [human, ai])
    checkpointer = FakeCheckpointer([latest, after_human, base])
    run_manager = FakeRunManager(
        [
            SimpleNamespace(run_id="run-latest", status=RunStatus.success, last_ai_message="different"),
        ]
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            _prepare_regenerate_payload(
                "thread-1",
                "ai-1",
                _request(checkpointer, FakeEventStore([]), run_manager=run_manager),
            )
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "Could not find source run for assistant message"


def test_prepare_regenerate_payload_requires_addressable_checkpoint_before_human():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(id="human-1", content="question")
    ai = AIMessage(id="ai-1", content="answer")
    latest = _checkpoint("ckpt-ai", [human, ai])
    checkpointer = FakeCheckpointer([latest])
    event_store = FakeEventStore(
        [
            {
                "run_id": "run-old",
                "event_type": "llm.ai.response",
                "category": "message",
                "content": {"id": "ai-1", "type": "ai", "content": "answer"},
                "metadata": {"caller": "lead_agent"},
            }
        ]
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_prepare_regenerate_payload("thread-1", "ai-1", _request(checkpointer, event_store)))

    assert exc.value.status_code == 409
    assert exc.value.detail == "Could not find an addressable checkpoint before the target user message"
    assert checkpointer.alist_limits == [400]


def test_prepare_regenerate_payload_reports_recent_checkpoint_scan_limit():
    from app.gateway.routers.thread_runs import _prepare_regenerate_payload

    human = HumanMessage(id="human-1", content="question")
    ai = AIMessage(id="ai-1", content="answer")
    latest = _checkpoint("ckpt-latest", [human, ai])
    history_without_human = [_checkpoint(f"ckpt-{index}", []) for index in range(201)]
    checkpointer = FakeCheckpointer(history_without_human, latest=latest)
    event_store = FakeEventStore(
        [
            {
                "run_id": "run-old",
                "event_type": "llm.ai.response",
                "category": "message",
                "content": {"id": "ai-1", "type": "ai", "content": "answer"},
                "metadata": {"caller": "lead_agent"},
            }
        ]
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_prepare_regenerate_payload("thread-1", "ai-1", _request(checkpointer, event_store)))

    assert exc.value.status_code == 409
    assert exc.value.detail == "Could not locate target user message in recent checkpoint history (limit=200)"
    assert checkpointer.alist_limits == [400]


def test_find_base_checkpoint_ignores_duration_only_checkpoints() -> None:
    from app.gateway.routers.thread_runs import _find_base_checkpoint_before_human

    human = HumanMessage(id="human-1", content="question")
    duration_checkpoints = [
        _checkpoint(
            f"duration-{index}",
            [],
            metadata={"writes": {"runtime_run_duration": {"run_ids": [f"run-{index}"]}}},
        )
        for index in range(200)
    ]
    base = _checkpoint("ckpt-base", [])
    after_human = _checkpoint("ckpt-human", [human])
    checkpointer = FakeCheckpointer([*duration_checkpoints, after_human, base])

    result = asyncio.run(_find_base_checkpoint_before_human("thread-1", "human-1", _request(checkpointer, FakeEventStore([]))))

    assert result is base
    assert checkpointer.alist_limits == [400]
