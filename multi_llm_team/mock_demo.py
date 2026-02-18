"""
Mock demo — runs the orchestrator WITHOUT real API calls.
Useful for testing the team structure and wiring without API keys.

Usage:
    python -m multi_llm_team.mock_demo
"""
from __future__ import annotations

from .agents.base_agent import AgentTask, AgentResult
from .orchestrator import MultiLLMOrchestrator


class _MockGemini:
    name = "Gemini"

    def execute(self, task: AgentTask) -> AgentResult:
        return AgentResult(
            agent_name="Gemini",
            task_id=task.task_id,
            role=task.role,
            output=(
                f"[MOCK Gemini — {task.role}]\n"
                f"Reviewed the task: '{task.prompt}'\n"
                "Findings:\n"
                "- The overall structure looks sound.\n"
                "- Consider adding input validation.\n"
                "- A few variable names could be more descriptive.\n"
                "- No obvious security issues detected."
            ),
            success=True,
            metadata={"model": "gemini-1.5-pro (mock)"},
        )


class _MockGPT:
    name = "GPT"

    def execute(self, task: AgentTask) -> AgentResult:
        return AgentResult(
            agent_name="GPT",
            task_id=task.task_id,
            role=task.role,
            output=(
                f"[MOCK GPT — {task.role}]\n"
                f"Implementing solution for: '{task.prompt}'\n\n"
                "```python\n"
                "# Refactored version (mock)\n"
                "def process_data(data: list) -> dict:\n"
                '    """Process and validate input data."""\n'
                "    if not data:\n"
                '        raise ValueError("data must not be empty")\n'
                "    return {i: v for i, v in enumerate(data)}\n"
                "```"
            ),
            success=True,
            metadata={"model": "gpt-4o (mock)"},
        )


def run_mock_demo() -> None:
    orchestrator = MultiLLMOrchestrator()
    # Monkey-patch with mocks so no API calls are made
    orchestrator.gemini = _MockGemini()  # type: ignore[assignment]
    orchestrator.gpt = _MockGPT()        # type: ignore[assignment]

    result = orchestrator.run(
        user_task="Refactor figure1_nature.py for better readability and add docstrings",
        context="(mock context — file contents would go here)",
    )
    print("\nDone. run_id:", result["run_id"])


if __name__ == "__main__":
    run_mock_demo()
