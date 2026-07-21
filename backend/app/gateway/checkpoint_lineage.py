"""Shared helpers for resolving replay checkpoints on one checkpoint lineage."""

from __future__ import annotations

import copy
from typing import Any


class CheckpointLineageError(RuntimeError):
    """Raised when a requested checkpoint ancestor cannot be resolved safely."""


def checkpoint_messages(checkpoint_tuple: Any) -> list[Any]:
    checkpoint = getattr(checkpoint_tuple, "checkpoint", None) or {}
    channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
    messages = channel_values.get("messages", []) if isinstance(channel_values, dict) else []
    return list(messages) if isinstance(messages, list) else []


def checkpoint_configurable(checkpoint_tuple: Any) -> dict[str, Any]:
    config = getattr(checkpoint_tuple, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    return dict(configurable) if isinstance(configurable, dict) else {}


def checkpoint_metadata(checkpoint_tuple: Any) -> dict[str, Any]:
    metadata = getattr(checkpoint_tuple, "metadata", None) or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def is_duration_only_checkpoint(checkpoint_tuple: Any) -> bool:
    writes = checkpoint_metadata(checkpoint_tuple).get("writes")
    return isinstance(writes, dict) and "runtime_run_duration" in writes


def _message_id(message: Any) -> str | None:
    value = getattr(message, "id", None)
    if value is None and isinstance(message, dict):
        value = message.get("id")
    return str(value) if value else None


def _checkpoint_identity(checkpoint_tuple: Any) -> tuple[str, str, str] | None:
    configurable = checkpoint_configurable(checkpoint_tuple)
    thread_id = configurable.get("thread_id")
    checkpoint_ns = configurable.get("checkpoint_ns", "")
    checkpoint_id = configurable.get("checkpoint_id")
    if not isinstance(thread_id, str) or not thread_id or not isinstance(checkpoint_id, str) or not checkpoint_id:
        return None
    return thread_id, str(checkpoint_ns or ""), checkpoint_id


async def find_checkpoint_before_message(
    checkpointer: Any,
    head_checkpoint: Any,
    message_id: str,
    *,
    max_depth: int,
) -> Any:
    """Walk one parent lineage and return the first checkpoint before ``message_id``.

    Following ``parent_config`` is important after a regenerate: a thread can contain
    sibling checkpoint branches, and a global time-ordered scan can otherwise select
    a checkpoint from the wrong branch. Duration-only metadata checkpoints do not
    represent an addressable conversation state and are skipped.
    """

    if message_id not in {_message_id(message) for message in checkpoint_messages(head_checkpoint)}:
        raise CheckpointLineageError("Target message is not present in the checkpoint head")

    current = head_checkpoint
    visited: set[tuple[str, str, str]] = set()
    current_identity = _checkpoint_identity(current)
    if current_identity is not None:
        visited.add(current_identity)

    for _ in range(max_depth):
        parent_config = getattr(current, "parent_config", None)
        if not isinstance(parent_config, dict):
            raise CheckpointLineageError("Checkpoint lineage ended before the target message")

        parent = await checkpointer.aget_tuple(parent_config)
        if parent is None:
            raise CheckpointLineageError("Checkpoint lineage references a missing parent")

        parent_identity = _checkpoint_identity(parent)
        if parent_identity is not None:
            if parent_identity in visited:
                raise CheckpointLineageError("Checkpoint lineage contains a cycle")
            visited.add(parent_identity)

        if is_duration_only_checkpoint(parent):
            current = parent
            continue

        parent_message_ids = {_message_id(message) for message in checkpoint_messages(parent)}
        if message_id not in parent_message_ids:
            if parent_identity is None:
                raise CheckpointLineageError("Checkpoint before the target message is not addressable")
            return parent
        current = parent

    raise CheckpointLineageError(f"Checkpoint lineage exceeded the scan limit ({max_depth})")


async def copy_checkpoint_to_thread(
    checkpointer: Any,
    source_checkpoint: Any,
    target_thread_id: str,
    *,
    metadata_updates: dict[str, Any],
    parent_config: dict[str, Any] | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Deep-copy a checkpoint into another thread and return its local config."""

    checkpoint = copy.deepcopy(getattr(source_checkpoint, "checkpoint", None) or {})
    metadata = checkpoint_metadata(source_checkpoint)
    metadata = copy.deepcopy(metadata)
    source_checkpoint_id = checkpoint_configurable(source_checkpoint).get("checkpoint_id")
    local_checkpoint_id = checkpoint_id or source_checkpoint_id
    if not isinstance(local_checkpoint_id, str) or not local_checkpoint_id:
        raise CheckpointLineageError("Source checkpoint is missing checkpoint_id")

    checkpoint["id"] = local_checkpoint_id
    metadata.update(metadata_updates)
    write_config = parent_config or {"configurable": {"thread_id": target_thread_id, "checkpoint_ns": ""}}
    new_versions = dict(checkpoint.get("channel_versions", {}) or {})
    return await checkpointer.aput(write_config, checkpoint, metadata, new_versions)
