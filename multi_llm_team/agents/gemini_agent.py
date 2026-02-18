"""Gemini agent - specializes in code review and analysis."""
import os
import json
from .base_agent import BaseAgent, AgentTask, AgentResult


class GeminiAgent(BaseAgent):
    """
    Agent powered by Google Gemini.
    Specialty: Code review, documentation analysis, multi-modal understanding.
    """

    def __init__(self, model: str = "gemini-1.5-pro"):
        super().__init__(
            name="Gemini",
            model=model,
            specialty=(
                "code review, documentation, multi-modal analysis, "
                "identifying edge cases and potential issues"
            ),
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai  # type: ignore

                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set."
                    )
                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=self._build_system_prompt(),
                )
            except ImportError as exc:
                raise ImportError(
                    "google-generativeai not installed. Run: pip install google-generativeai"
                ) from exc
        return self._client

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the task via Gemini API."""
        try:
            client = self._get_client()

            parts = []
            if task.context:
                parts.append(f"## Context\n{task.context}\n")
            if task.constraints:
                parts.append("## Constraints\n" + "\n".join(f"- {c}" for c in task.constraints))
            parts.append(f"## Task ({task.role})\n{task.prompt}")

            full_prompt = "\n\n".join(parts)
            response = client.generate_content(full_prompt)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                role=task.role,
                output=response.text,
                success=True,
                metadata={"model": self.model},
            )

        except Exception as exc:  # noqa: BLE001
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                role=task.role,
                output="",
                success=False,
                error=str(exc),
            )
