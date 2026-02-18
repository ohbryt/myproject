"""GPT agent - specializes in code generation and bug fixing."""
import os
from .base_agent import BaseAgent, AgentTask, AgentResult


class GPTAgent(BaseAgent):
    """
    Agent powered by OpenAI GPT.
    Specialty: Code generation, refactoring, test writing, bug fixing.
    """

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(
            name="GPT",
            model=model,
            specialty=(
                "code generation, refactoring, unit test writing, "
                "bug fixing, and implementation best practices"
            ),
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore

                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "OPENAI_API_KEY environment variable not set."
                    )
                self._client = OpenAI(api_key=api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai not installed. Run: pip install openai"
                ) from exc
        return self._client

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the task via OpenAI API."""
        try:
            client = self._get_client()

            user_parts = []
            if task.context:
                user_parts.append(f"## Context\n{task.context}")
            if task.constraints:
                user_parts.append(
                    "## Constraints\n" + "\n".join(f"- {c}" for c in task.constraints)
                )
            user_parts.append(f"## Task ({task.role})\n{task.prompt}")

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": "\n\n".join(user_parts)},
                ],
                temperature=0.2,
            )

            output = response.choices[0].message.content or ""
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                role=task.role,
                output=output,
                success=True,
                metadata={
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                },
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
