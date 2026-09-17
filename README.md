# study-helper

A multi-agent exam-prep tutor built on the OpenAI Agents SDK — a front desk routes
your question to a Math Tutor, a Science Tutor, or a Quiz Master, each with their
own tools and guardrails.

## Screenshots

![Math routing example](docs/screenshot1.png)

![Science tutor with web search](docs/screenshot2.png)

![Timed quiz with structured results](docs/screenshot3.png)

## Architecture

```
                        you
                         |
              Study Helper Front Desk          (orchestrator - routes, never teaches)
                         |
        +----------------+----------------+
        |                |                |
   Math Tutor      Science Tutor     Quiz Master
   calculate()      web_search()     one Q at a time
        |                |                |
        +-------- output guardrail -------+        (no leaked practice answers)
                         |
                   RunEvent stream                 (hooks -> terminal or web UI)
```

**Orchestrator.** Classifies each question and hands off. It teaches nothing; if a
question is neither math nor science it declines in one line itself. Handoffs use
`input_filter=remove_all_tools` — without it the receiving agent inherits the
completed transfer call, reads the tool phase as finished, and silently stops
using its own tools.

**Math Tutor** — `calculate()`, a function tool that walks an AST allow-list
(never `eval`). Instructed to make one call per intermediate result, so the steps
show real working rather than a single opaque answer.

**Science Tutor** — `web_search()`, `ddgs` with a Wikipedia fallback. For checking
facts, not for replacing explanations: conceptual "why" questions are answered
from the model's own knowledge.

**Quiz Master** — asks one question at a time under a total time budget. Numeric
answers are graded deterministically in Python; only open-ended ones fall through
to model judgment.

**Output guardrail.** Tutors are told to leave the "Check yourself" question
unsolved. They sometimes leak the answer anyway, so a guardrail detects it, trips
the tripwire, and hands the caller a repaired answer through `output_info`.

**Structured output.** The quiz returns a Pydantic `QuizResult`. The app computes
`score_percent`, `questions_answered` and `time_used_seconds`; the model supplies
only `weak_topics` and `verdict`, and is shown only the questions you got wrong.
A model outage costs you the prose, never the numbers.

**Hooks.** `RunHooks` emit a stream of `RunEvent` dataclasses. The terminal prints
them; the web UI renders tool calls as expandable steps. Neither interface parses
the other's strings.

## Install

Requires **Python >=3.13, <3.14**. Not 3.14: Chainlit calls `nest_asyncio.apply()`,
which breaks `asyncio.current_task()` on 3.14 and makes every static asset fail.

```bash
git clone https://github.com/m-shaheer313/study-helper.git
cd study-helper
uv sync
cp .env.example .env
```

Then edit `.env`. Exactly one provider block may be uncommented — `python-dotenv`
is last-wins, so several active blocks silently use the bottom one.

### The provider split

No free provider tested does everything well, so there are two model slots:

```bash
# Main agents: tutors, tools, handoffs
STUDY_HELPER_BASE_URL=https://api.groq.com/openai/v1
STUDY_HELPER_API_KEY=gsk_...
STUDY_HELPER_MODEL=openai/gpt-oss-120b

# Quiz agents only: these need structured output, and Groq cannot do it
STUDY_HELPER_STRUCTURED_BASE_URL=https://openrouter.ai/api/v1
STUDY_HELPER_STRUCTURED_API_KEY=sk-or-v1-...
STUDY_HELPER_STRUCTURED_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

| Provider | Tools | Handoffs | Multi-field JSON | Free limit |
|---|---|---|---|---|
| Groq `gpt-oss-120b` | yes | yes | **no** | 8000 tokens/min, 200k/day |
| OpenRouter Nemotron | **no** | **no** | yes | slow, occasional 503 |
| Gemini `3-flash-preview` | yes | yes | yes | **20 requests/day** |

Leave the three `STUDY_HELPER_STRUCTURED_*` variables unset and the quiz agents
fall back to the main model — the split disappears cleanly if you find one
provider that does both.

## Run

```bash
uv run study-helper                    # terminal
uv run chainlit run chainlit_app.py    # web UI, http://localhost:8000
```

Type `quiz` in either interface to start a timed quiz. Both build the same
orchestrator; no agent logic is duplicated between them, and tests enforce that.

## Tests

```bash
uv run python -m pytest            # 126 offline tests, no API key needed
uv run python -m pytest -m live    # 14 live tests, calls the real provider
```

Live tests are excluded by default. Free tiers rate-limit hard, so run them one
file at a time: `uv run python -m pytest -m live tests/test_quiz.py`.

Use `python -m pytest`, not `uv run pytest` — the console-script shim breaks after
a venv rebuild.

## Agents SDK concepts demonstrated

- **Multi-agent handoffs** — `handoff()` with `input_type`, `on_handoff`, and
  `input_filter`, plus a routing agent that declines out-of-scope work itself.
- **Function tools** — both kinds: `calculate()` for computation and
  `web_search()` for retrieval, with `failure_error_function` so a bad expression
  comes back to the model as a message instead of killing the run.
- **Output guardrails** — `@output_guardrail` detecting leaked answers, tripping a
  tripwire, and passing repaired text back through `output_info`.
- **Structured output** — Pydantic `output_type` for quiz questions, grading and
  results, using non-strict schema for provider compatibility.
- **Lifecycle hooks** — `RunHooks` across the whole run including handoffs,
  emitting structured events rather than printing.
- **Sessions** — `SQLiteSession` for per-conversation memory, isolated per browser
  session in the web UI.
- **Dual interface** — one system, a terminal client and a Chainlit web client.

## Known limitations

- **Personality never landed.** The tutors are instructed to be lightly witty. No
  free model tested delivers it — they obey the rule in the negative (never goofy,
  never sarcastic) but produce no actual voice. The scope, structure and
  formatting rules are followed; tone is the one instruction that needs a stronger
  model.
- **Groq's free tier interrupts long sessions.** Two limits: 8000 tokens per
  minute, and 200,000 per day. The Math Tutor's step-by-step decomposition can
  make seven tool calls for one question, so both are easier to reach than they
  look. The daily cap does not clear with a short wait.
- **`ddgs` is an unofficial DuckDuckGo scraper.** No API contract and no stability
  guarantee; it can rate-limit or break when the site changes. The Wikipedia
  fallback exists for exactly that, and a total failure returns
  `SEARCH_UNAVAILABLE` so the tutor answers from its own knowledge instead of
  erroring.
- **Python 3.14 is incompatible with Chainlit.** Chainlit's CLI calls
  `nest_asyncio.apply()`. On 3.14 that patching stops `asyncio.current_task()`
  from reporting the running task, so `sniffio` — which detects asyncio purely by
  that check — raises `AsyncLibraryNotFoundError`, anyio cannot select a backend,
  and `FileResponse`'s `anyio.to_thread.run_sync(os.stat, ...)` fails for every
  static asset. The page loads as a blank shell because the HTML itself is served
  from a string and never stats a file. Verified by A/B: same app, `apply()` the
  only difference — without it `current_task()` is a Task and assets return 200,
  with it `current_task()` is None and they return 500.
- **The terminal quiz timer is soft.** The budget is checked between questions,
  not during one, because a blocking `input()` cannot be cancelled cleanly. The
  web UI enforces it properly via `AskUserMessage(timeout=...)`.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Muhammad Shaheer.
