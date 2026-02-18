"""
Multi-LLM Team Orchestrator
============================
Claude acts as the leader. It decomposes a high-level task, assigns
sub-tasks to Gemini and GPT agents concurrently, and synthesizes the
results into a final answer.

Usage (standalone):
    python -m multi_llm_team.orchestrator "Refactor figure1_nature.py for readability"

Usage (from Claude Code Bash tool):
    python multi_llm_team/orchestrator.py "<your task>"
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .agents import AgentResult, AgentTask, GeminiAgent, GPTAgent

# ---------------------------------------------------------------------------
# Role definitions for the two specialist agents
# ---------------------------------------------------------------------------
GEMINI_ROLES = {
    "code_review": "Review the code for quality, correctness, edge cases, and potential issues.",
    "analysis": "Analyze the problem space and identify key considerations.",
    "documentation": "Review or generate clear, accurate documentation.",
    "security": "Identify security vulnerabilities or risks.",
}

GPT_ROLES = {
    "code_generation": "Generate clean, idiomatic implementation code.",
    "refactoring": "Refactor the existing code for better structure and readability.",
    "testing": "Write comprehensive unit tests covering edge cases.",
    "bug_fix": "Diagnose and fix the described bug with a minimal patch.",
}


# ---------------------------------------------------------------------------
# Decomposition helper (Claude does this interactively; here it's rule-based)
# ---------------------------------------------------------------------------

def decompose_task(user_task: str) -> tuple[str, str]:
    """
    Heuristically map a free-text task to (gemini_role, gpt_role).
    In real usage inside Claude Code, Claude itself decides the decomposition.
    """
    task_lower = user_task.lower()

    if any(w in task_lower for w in ("review", "check", "audit", "analyze", "analysis")):
        gemini_role = "code_review"
        gpt_role = "refactoring"
    elif any(w in task_lower for w in ("fix", "bug", "error", "broken", "crash")):
        gemini_role = "analysis"
        gpt_role = "bug_fix"
    elif any(w in task_lower for w in ("test", "spec", "unit")):
        gemini_role = "code_review"
        gpt_role = "testing"
    elif any(w in task_lower for w in ("doc", "readme", "comment", "explain")):
        gemini_role = "documentation"
        gpt_role = "code_generation"
    elif any(w in task_lower for w in ("refactor", "clean", "restructure")):
        gemini_role = "analysis"
        gpt_role = "refactoring"
    elif any(w in task_lower for w in ("secur", "vulnerab", "injection", "xss")):
        gemini_role = "security"
        gpt_role = "bug_fix"
    else:
        gemini_role = "code_review"
        gpt_role = "code_generation"

    return gemini_role, gpt_role


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MultiLLMOrchestrator:
    """
    Claude-led multi-LLM team orchestrator.

    Claude (this process) acts as the team leader:
      1. Decomposes the user task
      2. Dispatches sub-tasks to Gemini and GPT concurrently
      3. Collects and synthesizes results
    """

    def __init__(
        self,
        gemini_model: str = "gemini-1.5-pro",
        gpt_model: str = "gpt-4o",
        verbose: bool = True,
    ):
        self.gemini = GeminiAgent(model=gemini_model)
        self.gpt = GPTAgent(model=gpt_model)
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def run(
        self,
        user_task: str,
        context: Optional[str] = None,
        gemini_role: Optional[str] = None,
        gpt_role: Optional[str] = None,
    ) -> dict:
        """
        Execute the full multi-LLM workflow.

        Parameters
        ----------
        user_task   : High-level description of what the team should do.
        context     : Optional extra context (file contents, error messages, …).
        gemini_role : Override the auto-detected Gemini sub-task role.
        gpt_role    : Override the auto-detected GPT sub-task role.

        Returns
        -------
        dict with keys: gemini_result, gpt_result, synthesis
        """
        run_id = uuid.uuid4().hex[:8]
        self._log(f"\n{'='*60}")
        self._log(f"[Claude Leader] New team session: {run_id}")
        self._log(f"[Claude Leader] Task: {user_task}")
        self._log(f"{'='*60}\n")

        # --- Step 1: Decompose -----------------------------------------
        g_role, p_role = decompose_task(user_task)
        gemini_role = gemini_role or g_role
        gpt_role = gpt_role or p_role

        self._log(f"[Claude Leader] Assigning roles:")
        self._log(f"  Gemini -> {gemini_role}: {GEMINI_ROLES.get(gemini_role, gemini_role)}")
        self._log(f"  GPT    -> {gpt_role}: {GPT_ROLES.get(gpt_role, gpt_role)}")
        self._log("")

        gemini_task = AgentTask(
            task_id=f"{run_id}-gemini",
            role=gemini_role,
            prompt=user_task,
            context=context,
            constraints=["Be specific. Point to concrete lines/sections if possible."],
        )
        gpt_task = AgentTask(
            task_id=f"{run_id}-gpt",
            role=gpt_role,
            prompt=user_task,
            context=context,
            constraints=[
                "Provide working, ready-to-use code.",
                "Keep changes minimal and focused.",
            ],
        )

        # --- Step 2: Execute concurrently --------------------------------
        results: dict[str, AgentResult] = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.gemini.execute, gemini_task): "gemini",
                executor.submit(self.gpt.execute, gpt_task): "gpt",
            }
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                results[key] = result
                status = "OK" if result.success else f"ERROR: {result.error}"
                self._log(f"[Claude Leader] {result.agent_name} finished — {status}")

        # --- Step 3: Synthesize ------------------------------------------
        synthesis = self._synthesize(user_task, results["gemini"], results["gpt"])
        self._log("\n" + "="*60)
        self._log("[Claude Leader] SYNTHESIS")
        self._log("="*60)
        self._log(synthesis)

        return {
            "run_id": run_id,
            "task": user_task,
            "gemini_result": results["gemini"],
            "gpt_result": results["gpt"],
            "synthesis": synthesis,
        }

    # ------------------------------------------------------------------
    def _synthesize(
        self,
        task: str,
        gemini: AgentResult,
        gpt: AgentResult,
    ) -> str:
        """Claude (leader) synthesizes both agent outputs into a final answer."""
        lines = [
            f"## [Claude Leader] Synthesis — Task: {task}",
            "",
        ]

        if gemini.success:
            lines += [
                f"### Gemini ({gemini.role}) findings",
                gemini.output.strip(),
                "",
            ]
        else:
            lines += [f"### Gemini — failed: {gemini.error}", ""]

        if gpt.success:
            lines += [
                f"### GPT ({gpt.role}) output",
                gpt.output.strip(),
                "",
            ]
        else:
            lines += [f"### GPT — failed: {gpt.error}", ""]

        if gemini.success and gpt.success:
            lines += [
                "### Claude's integrated conclusion",
                (
                    "Both agents have completed their sub-tasks. "
                    "Gemini has provided review/analysis findings above. "
                    "GPT has provided implementation/code output above. "
                    "Review both sections and apply GPT's code changes "
                    "after verifying them against Gemini's feedback."
                ),
            ]
        elif not gemini.success and not gpt.success:
            lines += [
                "### Claude's integrated conclusion",
                "Both agents failed. Check API keys and retry.",
            ]
        else:
            lines += [
                "### Claude's integrated conclusion",
                "One agent failed. Use the successful agent's output and retry the failed one.",
            ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    task = " ".join(sys.argv[1:])

    # Optionally read context from stdin (piped file contents, etc.)
    context: Optional[str] = None
    if not sys.stdin.isatty():
        context = sys.stdin.read()

    orchestrator = MultiLLMOrchestrator()
    result = orchestrator.run(task, context=context)

    # Machine-readable summary to stdout for Claude Code to parse
    print("\n--- JSON SUMMARY ---")
    summary = {
        "run_id": result["run_id"],
        "task": result["task"],
        "gemini_success": result["gemini_result"].success,
        "gpt_success": result["gpt_result"].success,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
