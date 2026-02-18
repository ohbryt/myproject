# Claude Code — Multi-LLM Team Project

## Team Structure

This project uses a **3-LLM team** where Claude Code acts as the team leader:

| Role | Model | Specialty |
|---|---|---|
| **Leader (you)** | Claude (Sonnet/Opus) | Orchestration, synthesis, decision-making |
| **Gemini Agent** | gemini-1.5-pro | Code review, analysis, documentation, security |
| **GPT Agent** | gpt-4o | Code generation, refactoring, testing, bug fixes |

## How the team works

```
User Task
    │
    ▼
┌─────────────────────────────┐
│  Claude (Leader / Claude    │
│  Code session)              │
│  1. Understand task         │
│  2. Decompose into roles    │
│  3. Dispatch to agents      │
│  4. Synthesize results      │
└────────┬────────────┬───────┘
         │            │
    (concurrent)  (concurrent)
         │            │
    ┌────▼────┐  ┌────▼────┐
    │ Gemini  │  │   GPT   │
    │ Agent   │  │  Agent  │
    └─────────┘  └─────────┘
```

## Running a team task

### Via custom slash command (recommended inside Claude Code)
```
/team-task <your task>
```

### Via CLI
```bash
# Set API keys first
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run the team
python -m multi_llm_team.orchestrator "Refactor figure1_nature.py for readability"

# Pipe file context
cat figure1_nature.py | python -m multi_llm_team.orchestrator "Review this file"
```

### Via Python
```python
from multi_llm_team import MultiLLMOrchestrator

orchestrator = MultiLLMOrchestrator()
result = orchestrator.run(
    user_task="Improve the color palette in figure1_nature.py",
    context=open("figure1_nature.py").read(),
    gemini_role="code_review",   # optional override
    gpt_role="refactoring",      # optional override
)
print(result["synthesis"])
```

## Leader responsibilities (Claude's role)

As the team leader, Claude should:

1. **Decompose** complex tasks into focused sub-tasks for each specialist.
2. **Choose the right role** for each agent based on the task:
   - Gemini excels at reviewing existing code and finding issues.
   - GPT excels at generating new code and implementing fixes.
3. **Synthesize** both outputs — don't just concatenate. Identify agreements, conflicts, and the best path forward.
4. **Decide** when to iterate (e.g., send GPT's generated code back to Gemini for review).
5. **Report** clearly to the user what each agent contributed.

## Environment setup

```bash
pip install -r requirements_team.txt
export GEMINI_API_KEY="..."   # https://aistudio.google.com/app/apikey
export OPENAI_API_KEY="..."   # https://platform.openai.com/api-keys
```

## Project files

```
multi_llm_team/
├── __init__.py
├── orchestrator.py          # Main orchestration logic (Claude's control plane)
└── agents/
    ├── __init__.py
    ├── base_agent.py        # Abstract base class
    ├── gemini_agent.py      # Google Gemini agent
    └── gpt_agent.py        # OpenAI GPT agent

.claude/
└── commands/
    └── team-task.md        # /team-task slash command
```
