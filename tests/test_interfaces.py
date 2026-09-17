"""Both interfaces must share one system, not two copies of it."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHAINLIT_APP = ROOT / "chainlit_app.py"
SHARED = ROOT / "src" / "study_helper"


def _source(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


class TestNoDuplicatedAgentLogic:
    def test_chainlit_app_defines_no_agents_or_tools(self):
        tree = ast.parse(_source(CHAINLIT_APP))
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "Agent" not in calls, "the web UI must not define its own agents"
        assert "function_tool" not in calls, "the web UI must not define its own tools"

    def test_chainlit_app_has_no_instruction_prompts(self):
        # A long triple-quoted string here would mean prompt logic had leaked
        # out of roles/ into the UI layer.
        text = _source(CHAINLIT_APP)
        for marker in ("You are the", "SCOPE", "TONE", "STRUCTURE"):
            assert marker not in text, f"prompt text {marker!r} leaked into the UI"

    def test_both_interfaces_build_the_same_orchestrator(self):
        for path in (CHAINLIT_APP, SHARED / "cli.py"):
            assert "build_orchestrator" in _source(path), path.name


class TestSharedComponentsUntouchedByTheUI:
    """Step 6 was meant to be purely a new consumer of the existing system."""

    def test_ui_layer_imports_rather_than_reimplements(self):
        text = _source(CHAINLIT_APP)
        for shared in (
            "study_helper.roles.orchestrator",
            "study_helper.hooks",
            "study_helper.quiz_session",
            "study_helper.context",
        ):
            assert shared in text, f"{shared} should be imported, not reimplemented"

    def test_no_chainlit_dependency_in_the_shared_package(self):
        """The core must stay usable from the terminal and from tests."""
        for path in SHARED.rglob("*.py"):
            assert "import chainlit" not in _source(path), path
