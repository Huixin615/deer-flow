"""Apply skill ``allowed-tools`` only to skills active in lead-agent context."""

from __future__ import annotations

import asyncio
import logging
import posixpath
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.runtime.secret_context import SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY, read_slash_skill_source_path
from deerflow.skills.storage import get_or_new_skill_storage, get_or_new_user_skill_storage
from deerflow.skills.tool_policy import ALWAYS_AVAILABLE_BUILTIN_TOOL_NAMES, allowed_tool_names_for_skills
from deerflow.skills.types import Skill

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig
    from deerflow.skills.storage.skill_storage import SkillStorage

logger = logging.getLogger(__name__)

_POLICY_DECISION_VERSION = 1
_MISSING_POLICY_DECISION = object()


class SkillToolPolicyMiddleware(AgentMiddleware[AgentState]):
    """Restrict lead tools to declarations from slash/in-context skills.

    Merely enabling a skill makes it discoverable; it does not activate its
    authority policy. A skill becomes policy-active when the user slash-activates
    it for the run or after the model loads it into ``skill_context``.
    """

    def __init__(
        self,
        *,
        available_skills: set[str] | None = None,
        app_config: AppConfig | None = None,
        user_id: str | None = None,
    ) -> None:
        super().__init__()
        self._available_skills = set(available_skills) if available_skills is not None else None
        self._app_config = app_config
        self._user_id = user_id
        self._decision_owner_token = secrets.token_urlsafe(24)

    def _storage(self) -> SkillStorage:
        if self._user_id is not None:
            return get_or_new_user_skill_storage(self._user_id, app_config=self._app_config)
        if self._app_config is not None:
            return get_or_new_skill_storage(app_config=self._app_config)
        return get_or_new_skill_storage()

    @staticmethod
    def _active_paths(request: ModelRequest | ToolCallRequest) -> list[str]:
        paths: list[str] = []
        context = getattr(getattr(request, "runtime", None), "context", None)
        slash_path = read_slash_skill_source_path(context)
        if slash_path is not None:
            paths.append(slash_path)

        state = getattr(request, "state", None)
        if state is None:
            state = {}
        if isinstance(state, Mapping):
            entries = state.get("skill_context") or []
        elif hasattr(state, "skill_context"):
            entries = getattr(state, "skill_context") or []
        else:
            logger.warning("Unsupported agent state shape for skill tool policy: %s", type(state).__name__)
            entries = []
        if not isinstance(entries, (list, tuple)):
            logger.warning("Invalid skill_context shape for skill tool policy: %s", type(entries).__name__)
            entries = []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
        return paths

    def _active_skills_for_paths(self, paths: tuple[str, ...]) -> tuple[list[Skill], bool]:
        if not paths:
            return [], False

        try:
            storage = self._storage()
            skills = storage.load_skills(enabled_only=False)
            container_root = storage.get_container_root()
        except Exception:
            logger.exception("Failed to load active skills for allowed-tools policy")
            # A real active reference exists but cannot be authorized. Signal a
            # policy failure so callers retain only framework-safe tools.
            return [], True

        registry = {posixpath.normpath(skill.get_container_file_path(container_root)): skill for skill in skills}
        active: list[Skill] = []
        seen: set[str] = set()
        for path in paths:
            skill = registry.get(posixpath.normpath(path))
            if skill is None:
                logger.warning("Active skill path could not be resolved for allowed-tools policy: %s", path)
                continue
            if not skill.enabled or (self._available_skills is not None and skill.name not in self._available_skills):
                continue
            if skill.name in seen:
                continue
            seen.add(skill.name)
            active.append(skill)
        return active, False

    def _allowed_names_for_paths(self, paths: tuple[str, ...]) -> set[str] | None:
        active_skills, policy_failed = self._active_skills_for_paths(paths)
        if policy_failed:
            return set(ALWAYS_AVAILABLE_BUILTIN_TOOL_NAMES)
        allowed = allowed_tool_names_for_skills(active_skills)
        if allowed is None:
            return None
        return allowed | set(ALWAYS_AVAILABLE_BUILTIN_TOOL_NAMES)

    @staticmethod
    def _runtime_context(request: ModelRequest | ToolCallRequest) -> dict | None:
        context = getattr(getattr(request, "runtime", None), "context", None)
        return context if isinstance(context, dict) else None

    def _store_policy_decision(self, request: ModelRequest, paths: tuple[str, ...], allowed: set[str] | None) -> None:
        context = self._runtime_context(request)
        if context is not None:
            context[SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY] = {
                "version": _POLICY_DECISION_VERSION,
                "owner_token": self._decision_owner_token,
                "active_paths": list(paths),
                "allowed_names": None if allowed is None else sorted(allowed),
            }

    def _read_policy_decision(self, context: dict | None, paths: tuple[str, ...]) -> set[str] | None | object:
        if context is None:
            return _MISSING_POLICY_DECISION
        decision = context.get(SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY)
        if not isinstance(decision, dict):
            return _MISSING_POLICY_DECISION
        if type(decision.get("version")) is not int or decision["version"] != _POLICY_DECISION_VERSION:
            return _MISSING_POLICY_DECISION
        if not isinstance(decision.get("owner_token"), str) or decision["owner_token"] != self._decision_owner_token:
            return _MISSING_POLICY_DECISION
        stored_paths = decision.get("active_paths")
        if not isinstance(stored_paths, list) or not all(isinstance(path, str) for path in stored_paths) or tuple(stored_paths) != paths:
            return _MISSING_POLICY_DECISION
        allowed = decision.get("allowed_names")
        if allowed is None:
            return None
        if not isinstance(allowed, list) or not all(isinstance(name, str) for name in allowed):
            return _MISSING_POLICY_DECISION
        return set(allowed)

    def _allowed_names(self, request: ModelRequest | ToolCallRequest) -> set[str] | None:
        paths = tuple(self._active_paths(request))
        context = self._runtime_context(request)
        decision = self._read_policy_decision(context, paths)
        if decision is not _MISSING_POLICY_DECISION:
            return decision
        return self._allowed_names_for_paths(paths)

    def _filter_model_request(
        self,
        request: ModelRequest,
        *,
        paths: tuple[str, ...] | None = None,
        refresh_decision: bool = False,
    ) -> ModelRequest:
        resolved_paths = tuple(self._active_paths(request)) if paths is None else paths
        allowed = self._allowed_names_for_paths(resolved_paths) if refresh_decision else self._allowed_names(request)
        if refresh_decision:
            self._store_policy_decision(request, resolved_paths, allowed)
        if allowed is None:
            return request
        tools = [tool for tool in request.tools if getattr(tool, "name", None) in allowed]
        if len(tools) < len(request.tools):
            logger.debug("Skill policy filtered %d lead tool schema(s)", len(request.tools) - len(tools))
        return request.override(tools=tools)

    def _blocked_tool_message(self, request: ToolCallRequest) -> ToolMessage | None:
        allowed = self._allowed_names(request)
        name = str(request.tool_call.get("name") or "")
        if allowed is None or not name or name in allowed:
            return None
        return ToolMessage(
            content=f"Error: Tool '{name}' is not allowed by the active skill policy.",
            tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"),
            name=name,
            status="error",
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        paths = tuple(self._active_paths(request))
        return handler(self._filter_model_request(request, paths=paths, refresh_decision=True))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        paths = tuple(self._active_paths(request))
        if not paths:
            self._store_policy_decision(request, paths, None)
            return await handler(request)
        filtered = await asyncio.to_thread(
            self._filter_model_request,
            request,
            paths=paths,
            refresh_decision=True,
        )
        return await handler(filtered)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if not self._active_paths(request):
            return handler(request)
        blocked = self._blocked_tool_message(request)
        if blocked is not None:
            return blocked
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if not self._active_paths(request):
            return await handler(request)
        blocked = await asyncio.to_thread(self._blocked_tool_message, request)
        if blocked is not None:
            return blocked
        return await handler(request)
