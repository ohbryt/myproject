# /team-task — Multi-LLM Team Task

Invoke the multi-LLM team to collaborate on a task.
Claude (you) acts as the **leader**. Gemini handles review/analysis; GPT handles implementation.

## Usage

```
/team-task <task description>
```

## What to do when this command is invoked

1. **Understand** the user's task from `$ARGUMENTS`.
2. **Decide** which role each agent should take:
   - Gemini: `code_review` | `analysis` | `documentation` | `security`
   - GPT:    `code_generation` | `refactoring` | `testing` | `bug_fix`
3. **Run** the orchestrator via Bash:

```bash
python -m multi_llm_team.orchestrator "$ARGUMENTS"
```

Or with file context piped in:

```bash
cat <relevant_file> | python -m multi_llm_team.orchestrator "$ARGUMENTS"
```

4. **Parse** the JSON summary at the end of the output.
5. **Synthesize** the Gemini and GPT outputs and present a unified answer to the user.
6. If either agent failed, diagnose the error (likely missing API key) and inform the user.

## Required environment variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key for Gemini |
| `OPENAI_API_KEY` | OpenAI API key for GPT |

## Example invocations

```
/team-task Review and improve figure1_nature.py
/team-task Fix the color palette bug in the bar chart
/team-task Write unit tests for the data loading section
/team-task Add docstrings to all functions in figure1_nature.py
```
