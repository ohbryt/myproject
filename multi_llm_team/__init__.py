"""
multi_llm_team
==============
A multi-LLM team orchestration system for Claude Code.

Claude acts as the team leader and orchestrates:
  - Gemini (Google) — code review, analysis, documentation
  - GPT   (OpenAI)  — code generation, refactoring, testing, bug fixes
"""
from .orchestrator import MultiLLMOrchestrator
from .agents import GeminiAgent, GPTAgent, AgentTask, AgentResult

__all__ = [
    "MultiLLMOrchestrator",
    "GeminiAgent",
    "GPTAgent",
    "AgentTask",
    "AgentResult",
]
