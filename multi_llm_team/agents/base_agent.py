"""Base agent class for all LLM agents in the team."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentTask:
    """A task assigned to an agent."""
    task_id: str
    role: str           # e.g., "code_review", "code_generation", "analysis"
    prompt: str
    context: Optional[str] = None
    constraints: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result returned by an agent."""
    agent_name: str
    task_id: str
    role: str
    output: str
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all LLM agents."""

    def __init__(self, name: str, model: str, specialty: str):
        self.name = name
        self.model = model
        self.specialty = specialty

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task and return the result."""
        pass

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.name}, a specialist AI agent in a multi-LLM team. "
            f"Your specialty: {self.specialty}. "
            "Be concise, precise, and focus on your area of expertise. "
            "Format your response clearly so the team leader (Claude) can synthesize it."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model!r})"
