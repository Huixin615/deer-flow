import asyncio
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from deerflow.runtime.secret_context import _SLASH_SECRET_SOURCE_KEY
from deerflow.skills.types import Skill, SkillCategory


class NamedTool:
    def __init__(self, name: str):
        self.name = name


class ModelRequestStub:
    def __init__(self, tools, *, state=None, context=None, messages=None):
        self.tools = tools
        self.state = state or {}
        self.runtime = SimpleNamespace(context=context or {})
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
        self.runtime = SimpleNamespace(context=context or {})


class StorageStub:
    def __init__(self, skills):
        self._skills = skills

    def load_skills(self, *, enabled_only=False):
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


def test_slash_activated_skill_filters_first_model_call_and_task():
    skill = _skill("reviewer", ["review_skill_package"])
    context = {_SLASH_SECRET_SOURCE_KEY: {"path": skill.get_container_file_path()}}
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
