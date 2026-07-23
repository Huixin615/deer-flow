from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest
from langgraph.config import get_stream_writer
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Interrupt

from deerflow.utils import custom_events as custom_events_module
from deerflow.utils.custom_events import aemit_custom_event, emit_custom_event


class _State(TypedDict):
    value: int


def _compile_graph(node):
    builder = StateGraph(_State)
    builder.add_node("emit", node)
    builder.add_edge(START, "emit")
    builder.add_edge("emit", END)
    return builder.compile()


def _sync_node(state: _State) -> _State:
    payload = {"type": "sync_probe", "value": state["value"]}
    emit_custom_event(payload, writer=get_stream_writer())
    return state


async def _async_node(state: _State) -> _State:
    payload = {"type": "async_probe", "value": state["value"]}
    await aemit_custom_event(payload, writer=get_stream_writer())
    return state


async def _custom_events(graph) -> list[dict]:
    return [chunk async for chunk in graph.astream({"value": 7}, stream_mode="custom")]


async def _astream_events(graph) -> list[dict]:
    return [event async for event in graph.astream_events({"value": 7}, version="v2") if event["event"] == "on_custom_event"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("node", "event_name"),
    [
        (_sync_node, "sync_probe"),
        (_async_node, "async_probe"),
    ],
)
async def test_custom_event_is_emitted_once_to_each_streaming_api(node, event_name):
    graph = _compile_graph(node)

    custom_chunks = await _custom_events(graph)
    callback_events = await _astream_events(graph)

    expected = {"type": event_name, "value": 7}
    assert custom_chunks == [expected]
    assert len(callback_events) == 1
    assert callback_events[0]["name"] == event_name
    assert callback_events[0]["data"] == expected


def test_sync_dispatch_failure_does_not_break_writer(monkeypatch):
    payload = {"type": "sync_probe", "value": 1}
    written: list[dict] = []

    def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError("callback failed")

    monkeypatch.setattr(custom_events_module, "dispatch_custom_event", fail_dispatch)

    emit_custom_event(payload, writer=written.append)

    assert written == [payload]


def test_sync_dispatch_without_parent_run_does_not_break_writer():
    payload = {"type": "sync_probe", "value": 1}
    written: list[dict] = []

    emit_custom_event(payload, writer=written.append)

    assert written == [payload]


@pytest.mark.anyio
async def test_async_dispatch_failure_does_not_break_writer(monkeypatch):
    payload = {"type": "async_probe", "value": 1}
    written: list[dict] = []

    async def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError("callback failed")

    monkeypatch.setattr(custom_events_module, "adispatch_custom_event", fail_dispatch)

    await aemit_custom_event(payload, writer=written.append)

    assert written == [payload]


@pytest.mark.anyio
async def test_async_dispatch_without_parent_run_does_not_break_writer():
    payload = {"type": "async_probe", "value": 1}
    written: list[dict] = []

    await aemit_custom_event(payload, writer=written.append)

    assert written == [payload]


def test_missing_event_type_preserves_writer_and_skips_dispatch(monkeypatch):
    payload = {"value": 1}
    written: list[dict] = []
    dispatched: list[tuple] = []

    monkeypatch.setattr(custom_events_module, "dispatch_custom_event", lambda *args, **kwargs: dispatched.append((args, kwargs)))

    emit_custom_event(payload, writer=written.append)

    assert written == [payload]
    assert dispatched == []


def test_writer_failure_propagates_before_dispatch(monkeypatch):
    dispatched: list[tuple] = []

    def fail_writer(_payload):
        raise RuntimeError("writer failed")

    monkeypatch.setattr(custom_events_module, "dispatch_custom_event", lambda *args, **kwargs: dispatched.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="writer failed"):
        emit_custom_event({"type": "sync_probe"}, writer=fail_writer)

    assert dispatched == []


@pytest.mark.anyio
async def test_async_writer_failure_propagates_before_dispatch(monkeypatch):
    dispatched: list[tuple] = []

    def fail_writer(_payload):
        raise RuntimeError("writer failed")

    async def record_dispatch(*args, **kwargs):
        dispatched.append((args, kwargs))

    monkeypatch.setattr(custom_events_module, "adispatch_custom_event", record_dispatch)

    with pytest.raises(RuntimeError, match="writer failed"):
        await aemit_custom_event({"type": "async_probe"}, writer=fail_writer)

    assert dispatched == []


@pytest.mark.anyio
async def test_async_cancellation_is_not_swallowed(monkeypatch):
    async def cancel_dispatch(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(custom_events_module, "adispatch_custom_event", cancel_dispatch)

    with pytest.raises(asyncio.CancelledError):
        await aemit_custom_event({"type": "async_probe"}, writer=lambda _payload: None)


@pytest.mark.parametrize("async_dispatch", [False, True])
def test_langgraph_control_flow_is_not_swallowed(monkeypatch, async_dispatch):
    control_flow = GraphInterrupt((Interrupt(value="pause"),))

    if async_dispatch:

        async def interrupt_dispatch(*_args, **_kwargs):
            raise control_flow

        monkeypatch.setattr(custom_events_module, "adispatch_custom_event", interrupt_dispatch)

        with pytest.raises(GraphInterrupt) as raised:
            asyncio.run(aemit_custom_event({"type": "async_probe"}, writer=lambda _payload: None))
    else:

        def interrupt_dispatch(*_args, **_kwargs):
            raise control_flow

        monkeypatch.setattr(custom_events_module, "dispatch_custom_event", interrupt_dispatch)

        with pytest.raises(GraphInterrupt) as raised:
            emit_custom_event({"type": "sync_probe"}, writer=lambda _payload: None)

    assert raised.value is control_flow
