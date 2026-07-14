import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

from deerflow.runtime.secret_context import SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY, write_slash_skill_source_path
from deerflow.skills.types import Skill, SkillCategory


class NamedTool:
    def __init__(self, name: str):
        self.name = name


class ModelRequestStub:
    def __init__(self, tools, *, state=None, context=None, messages=None):
        self.tools = tools
        self.state = state or {}
        self.runtime = SimpleNamespace(context={} if context is None else context)
        self.messages = messages or []

    def override(self, **updates):
        return ModelRequestStub(
            updates.get("tools", self.tools),
            state=updates.get("state", self.state),
            context=self.runtime.context,
            messages=updates.get("messages", self.messages),
        )


class ToolRequestStub:
    def __init__(self, name: str, *, state=None, context=None):
        self.tool_call = {"name": name, "id": "call-1", "args": {}}
        self.state = state or {}
        self.runtime = SimpleNamespace(context={} if context is None else context)


class StorageStub:
    def __init__(self, skills):
        self._skills = skills
        self.load_calls = 0

    def load_skills(self, *, enabled_only=False):
        self.load_calls += 1
        return [skill for skill in self._skills if skill.enabled or not enabled_only]

    def get_container_root(self):
        return "/mnt/skills"


def _skill(name: str, allowed_tools, *, enabled=True):
    skill_dir = Path(f"/tmp/skills/public/{name}")
    return Skill(
        name=name,
        description=f"Description for {name}",
        license="MIT",
        skill_dir=skill_dir,
        skill_file=skill_dir / "SKILL.md",
        relative_path=Path(name),
        category=SkillCategory.PUBLIC,
        allowed_tools=None if allowed_tools is None else tuple(allowed_tools),
        enabled=enabled,
    )


def _middleware(skills, *, available_skills=None):
    from deerflow.agents.middlewares.skill_tool_policy_middleware import SkillToolPolicyMiddleware

    middleware = SkillToolPolicyMiddleware(available_skills=available_skills)
    middleware._storage = lambda: StorageStub(skills)
    return middleware


def _tool_names(request):
    return [tool.name for tool in request.tools]


def test_passive_enabled_skill_does_not_filter_lead_tools():
    middleware = _middleware([_skill("reviewer", ["review_skill_package"])])
    request = ModelRequestStub([NamedTool("task"), NamedTool("web_search"), NamedTool("review_skill_package")])

    filtered = middleware._filter_model_request(request)

    assert _tool_names(filtered) == ["task", "web_search", "review_skill_package"]


def test_sync_passive_model_call_skips_storage():
    middleware = _middleware([])

    def fail_storage():
        raise AssertionError("passive model calls must not load skill storage")

    middleware._storage = fail_storage
    request = ModelRequestStub([NamedTool("task")])

    assert middleware.wrap_model_call(request, lambda model_request: model_request) is request


def test_async_passive_model_call_skips_storage_and_thread_offload():
    middleware = _middleware([])

    def fail_storage():
        raise AssertionError("passive model calls must not load skill storage")

    middleware._storage = fail_storage
    request = ModelRequestStub([NamedTool("task")])

    async def handler(model_request):
        return model_request

    assert asyncio.run(middleware.awrap_model_call(request, handler)) is request


def test_slash_activated_skill_filters_first_model_call_and_task():
    skill = _skill("reviewer", ["review_skill_package"])
    context = {}
    write_slash_skill_source_path(context, skill.get_container_file_path())
    middleware = _middleware([skill])
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("read_file"), NamedTool("review_skill_package")],
        context=context,
    )

    filtered = middleware._filter_model_request(request)

    assert _tool_names(filtered) == ["read_file", "review_skill_package"]


def test_slash_activation_and_policy_compose_on_the_same_model_call(monkeypatch):
    from deerflow.agents.middlewares.skill_activation_middleware import SkillActivationMiddleware, _Activation, _ActivationResolution

    skill = _skill("reviewer", ["review_skill_package"])
    activation = _Activation(
        skill_name=skill.name,
        category="public",
        container_file_path=skill.get_container_file_path(),
        skill_content="# Reviewer",
        content_hash="abc",
        remaining_text="review this",
        editable=False,
    )
    activation_middleware = SkillActivationMiddleware()
    monkeypatch.setattr(activation_middleware, "_resolve_activation", lambda _: _ActivationResolution(activation=activation))
    policy_middleware = _middleware([skill])
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("read_file"), NamedTool("review_skill_package")],
        messages=[HumanMessage(content="/reviewer review this")],
    )

    filtered = activation_middleware.wrap_model_call(
        request,
        lambda activated: policy_middleware.wrap_model_call(activated, lambda policy_request: policy_request),
    )

    assert _tool_names(filtered) == ["read_file", "review_skill_package"]


def test_loaded_skill_context_filters_follow_up_model_calls():
    skill = _skill("restricted", ["web_search"])
    middleware = _middleware([skill])
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("read_file"), NamedTool("web_search")],
        state={"skill_context": [{"name": skill.name, "path": skill.get_container_file_path()}]},
    )

    filtered = middleware._filter_model_request(request)

    assert _tool_names(filtered) == ["read_file", "web_search"]


def test_active_skill_union_and_legacy_semantics_are_preserved():
    restricted = _skill("restricted", ["web_search"])
    second = _skill("second", ["bash"])
    legacy = _skill("legacy", None)
    middleware = _middleware([restricted, second, legacy])
    state = {
        "skill_context": [
            {"path": restricted.get_container_file_path()},
            {"path": second.get_container_file_path()},
            {"path": legacy.get_container_file_path()},
        ]
    }
    request = ModelRequestStub([NamedTool("task"), NamedTool("bash"), NamedTool("web_search")], state=state)

    filtered = middleware._filter_model_request(request)

    assert _tool_names(filtered) == ["bash", "web_search"]


def test_only_legacy_active_skill_preserves_all_tools():
    legacy = _skill("legacy", None)
    middleware = _middleware([legacy])
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("bash")],
        state={"skill_context": [{"path": legacy.get_container_file_path()}]},
    )

    assert _tool_names(middleware._filter_model_request(request)) == ["task", "bash"]


def test_explicit_empty_allowed_tools_keeps_only_framework_tools():
    restricted = _skill("restricted", [])
    middleware = _middleware([restricted])
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("read_file"), NamedTool("review_skill_package")],
        state={"skill_context": [{"path": restricted.get_container_file_path()}]},
    )

    assert _tool_names(middleware._filter_model_request(request)) == ["read_file", "review_skill_package"]


def test_custom_agent_allowlist_ignores_out_of_scope_skill_context():
    restricted = _skill("restricted", ["web_search"])
    middleware = _middleware([restricted], available_skills={"other"})
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("web_search")],
        state={"skill_context": [{"path": restricted.get_container_file_path()}]},
    )

    assert _tool_names(middleware._filter_model_request(request)) == ["task", "web_search"]


def test_unauthorized_tool_execution_is_blocked():
    restricted = _skill("restricted", ["web_search"])
    middleware = _middleware([restricted])
    request = ToolRequestStub(
        "task",
        state={"skill_context": [{"path": restricted.get_container_file_path()}]},
    )

    result = middleware.wrap_tool_call(request, lambda _: "executed")

    assert result.status == "error"
    assert result.name == "task"
    assert "not allowed" in result.content


def test_allowed_tool_execution_reaches_handler():
    restricted = _skill("restricted", ["web_search"])
    middleware = _middleware([restricted])
    request = ToolRequestStub(
        "web_search",
        state={"skill_context": [{"path": restricted.get_container_file_path()}]},
    )

    assert middleware.wrap_tool_call(request, lambda _: "executed") == "executed"


def test_async_unauthorized_tool_execution_is_blocked():
    restricted = _skill("restricted", ["web_search"])
    middleware = _middleware([restricted])
    request = ToolRequestStub(
        "task",
        state={"skill_context": [{"path": restricted.get_container_file_path()}]},
    )

    async def handler(_):
        return "executed"

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert result.status == "error"
    assert result.name == "task"


def test_unknown_skill_context_path_is_skipped_while_resolvable_skills_apply():
    restricted = _skill("restricted", ["web_search"])
    middleware = _middleware([restricted])
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("read_file"), NamedTool("web_search")],
        state={
            "skill_context": [
                {"path": "/mnt/skills/public/missing/SKILL.md"},
                {"path": restricted.get_container_file_path()},
            ]
        },
    )

    assert _tool_names(middleware._filter_model_request(request)) == ["read_file", "web_search"]


def test_async_passive_tool_call_skips_storage_and_thread_offload():
    middleware = _middleware([])

    def fail_storage():
        raise AssertionError("passive tool calls must not load skill storage")

    middleware._storage = fail_storage
    request = ToolRequestStub("task")

    async def handler(_):
        return "executed"

    assert asyncio.run(middleware.awrap_tool_call(request, handler)) == "executed"


def test_sync_passive_tool_call_skips_policy_resolution():
    middleware = _middleware([])
    request = ToolRequestStub("task")
    middleware._blocked_tool_message = MagicMock(side_effect=AssertionError("passive tool calls must bypass policy resolution"))

    assert middleware.wrap_tool_call(request, lambda _: "executed") == "executed"
    middleware._blocked_tool_message.assert_not_called()


def test_tool_calls_reuse_the_current_model_step_policy_decision():
    restricted = _skill("restricted", ["web_search"])
    storage = StorageStub([restricted])
    middleware = _middleware([])
    middleware._storage = lambda: storage
    context = {}
    state = {"skill_context": [{"path": restricted.get_container_file_path()}]}
    model_request = ModelRequestStub(
        [NamedTool("task"), NamedTool("web_search")],
        state=state,
        context=context,
    )

    filtered = middleware.wrap_model_call(model_request, lambda request: request)
    assert _tool_names(filtered) == ["web_search"]

    for _ in range(3):
        tool_request = ToolRequestStub("web_search", state=state, context=context)
        assert middleware.wrap_tool_call(tool_request, lambda _: "executed") == "executed"

    assert storage.load_calls == 1


def test_async_tool_calls_reuse_the_current_model_step_policy_decision():
    restricted = _skill("restricted", ["web_search"])
    storage = StorageStub([restricted])
    middleware = _middleware([])
    middleware._storage = lambda: storage
    context = {}
    state = {"skill_context": [{"path": restricted.get_container_file_path()}]}
    model_request = ModelRequestStub(
        [NamedTool("task"), NamedTool("web_search")],
        state=state,
        context=context,
    )

    async def go():
        filtered = await middleware.awrap_model_call(model_request, lambda request: asyncio.sleep(0, result=request))
        assert _tool_names(filtered) == ["web_search"]

        async def execute(_):
            return "executed"

        results = await asyncio.gather(*(middleware.awrap_tool_call(ToolRequestStub("web_search", state=state, context=context), execute) for _ in range(3)))
        assert results == ["executed", "executed", "executed"]

    asyncio.run(go())
    assert storage.load_calls == 1


def test_real_model_and_tool_requests_share_the_model_step_policy_decision():
    restricted = _skill("restricted", ["web_search"])
    storage = StorageStub([restricted])
    middleware = _middleware([])
    middleware._storage = lambda: storage
    context = {}
    state = {
        "messages": [],
        "skill_context": [{"path": restricted.get_container_file_path()}],
    }
    model_request = ModelRequest(
        model=MagicMock(),
        messages=[],
        tools=[NamedTool("task"), NamedTool("web_search")],
        state=state,
        runtime=Runtime(context=context),
    )

    filtered = middleware.wrap_model_call(model_request, lambda request: request)
    assert _tool_names(filtered) == ["web_search"]

    tool_runtime = ToolRuntime(
        state=state,
        context=context,
        config={},
        stream_writer=lambda _: None,
        tools=[],
        tool_call_id="call-1",
        store=None,
    )
    tool_request = ToolCallRequest(
        tool_call={"name": "web_search", "args": {}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state=state,
        runtime=tool_runtime,
    )

    assert middleware.wrap_tool_call(tool_request, lambda _: "executed") == "executed"
    assert storage.load_calls == 1


def test_policy_decision_is_json_safe_and_survives_round_trip():
    restricted = _skill("restricted", ["web_search"])
    storage = StorageStub([restricted])
    middleware = _middleware([])
    middleware._storage = lambda: storage
    state = {"skill_context": [{"path": restricted.get_container_file_path()}]}
    context = {}
    model_request = ModelRequestStub([NamedTool("web_search")], state=state, context=context)

    middleware.wrap_model_call(model_request, lambda request: request)
    round_tripped = json.loads(json.dumps(context))
    tool_request = ToolRequestStub("web_search", state=state, context=round_tripped)

    assert middleware.wrap_tool_call(tool_request, lambda _: "executed") == "executed"
    assert storage.load_calls == 1


def test_forged_or_malformed_policy_decisions_fall_back_to_live_resolution():
    restricted = _skill("restricted", ["web_search"])
    malformed_decisions = [
        None,
        [],
        {"version": 999, "owner_token": "forged", "active_paths": [restricted.get_container_file_path()], "allowed_names": ["task"]},
        {"version": True, "owner_token": "forged", "active_paths": [restricted.get_container_file_path()], "allowed_names": ["task"]},
        {"version": 1, "owner_token": "forged", "active_paths": [restricted.get_container_file_path()], "allowed_names": ["task"]},
        {"version": 1, "owner_token": "forged", "active_paths": "not-a-list", "allowed_names": ["task"]},
    ]

    for decision in malformed_decisions:
        storage = StorageStub([restricted])
        middleware = _middleware([])
        middleware._storage = lambda storage=storage: storage
        context = {SKILL_TOOL_POLICY_DECISION_CONTEXT_KEY: decision}
        request = ToolRequestStub(
            "task",
            state={"skill_context": [{"path": restricted.get_container_file_path()}]},
            context=context,
        )

        result = middleware.wrap_tool_call(request, lambda _: "executed")

        assert result.status == "error"
        assert storage.load_calls == 1


def test_policy_decision_path_mismatch_falls_back_to_live_resolution():
    first = _skill("first", ["web_search"])
    second = _skill("second", ["bash"])
    storage = StorageStub([first, second])
    middleware = _middleware([])
    middleware._storage = lambda: storage
    context = {}
    first_state = {"skill_context": [{"path": first.get_container_file_path()}]}
    second_state = {"skill_context": [{"path": second.get_container_file_path()}]}

    middleware.wrap_model_call(ModelRequestStub([NamedTool("web_search")], state=first_state, context=context), lambda request: request)
    result = middleware.wrap_tool_call(ToolRequestStub("web_search", state=second_state, context=context), lambda _: "executed")

    assert result.status == "error"
    assert storage.load_calls == 2


def test_active_paths_support_attribute_based_state():
    restricted = _skill("restricted", ["web_search"])
    middleware = _middleware([restricted])
    state = SimpleNamespace(skill_context=[{"path": restricted.get_container_file_path()}])
    request = ModelRequestStub([NamedTool("task"), NamedTool("web_search")], state=state)

    assert _tool_names(middleware._filter_model_request(request)) == ["web_search"]


def test_active_paths_support_falsey_attribute_based_state():
    class FalseyState:
        skill_context = []

        def __bool__(self):
            return False

    restricted = _skill("restricted", ["web_search"])
    state = FalseyState()
    state.skill_context = [{"path": restricted.get_container_file_path()}]
    middleware = _middleware([restricted])
    request = ModelRequestStub([NamedTool("task"), NamedTool("web_search")])
    request.state = state

    assert _tool_names(middleware._filter_model_request(request)) == ["web_search"]


def test_unknown_state_shape_is_logged_instead_of_silently_ignored(caplog):
    middleware = _middleware([])
    request = ModelRequestStub([NamedTool("task")], state=object())

    assert _tool_names(middleware._filter_model_request(request)) == ["task"]
    assert "Unsupported agent state shape" in caplog.text


def test_next_model_call_refreshes_the_policy_decision():
    restricted = _skill("restricted", ["web_search"])
    storage = StorageStub([restricted])
    middleware = _middleware([])
    middleware._storage = lambda: storage
    context = {}
    state = {"skill_context": [{"path": restricted.get_container_file_path()}]}

    first = ModelRequestStub([NamedTool("bash"), NamedTool("web_search")], state=state, context=context)
    assert _tool_names(middleware.wrap_model_call(first, lambda request: request)) == ["web_search"]

    storage._skills = [_skill("restricted", ["bash"])]
    second = ModelRequestStub([NamedTool("bash"), NamedTool("web_search")], state=state, context=context)
    assert _tool_names(middleware.wrap_model_call(second, lambda request: request)) == ["bash"]
    assert storage.load_calls == 2


def test_tool_call_without_matching_model_decision_revalidates_registry():
    restricted = _skill("restricted", ["web_search"])
    storage = StorageStub([restricted])
    middleware = _middleware([])
    middleware._storage = lambda: storage
    request = ToolRequestStub(
        "task",
        state={"skill_context": [{"path": restricted.get_container_file_path()}]},
        context={},
    )

    result = middleware.wrap_tool_call(request, lambda _: "executed")

    assert result.status == "error"
    assert storage.load_calls == 1


def test_active_policy_load_failure_fails_closed_to_framework_tools():
    middleware = _middleware([])

    def fail_storage():
        raise RuntimeError("storage unavailable")

    middleware._storage = fail_storage
    request = ModelRequestStub(
        [NamedTool("task"), NamedTool("read_file"), NamedTool("review_skill_package")],
        state={"skill_context": [{"path": "/mnt/skills/public/restricted/SKILL.md"}]},
    )

    assert _tool_names(middleware._filter_model_request(request)) == ["read_file", "review_skill_package"]
