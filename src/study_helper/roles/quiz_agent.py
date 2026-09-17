"""The Quiz Agent: asks one question at a time, judges open-ended answers.

Four configurations of one persona. The plain agent is what the orchestrator
hands off to in normal chat; the three structured ones are used by quiz mode,
each with a narrow output_type so the app gets a typed object back instead of
prose it would have to parse.

Groq rejects the SDK's default strict JSON schema outright (400
json_validate_failed, even for a single string field), so every structured
output here is non-strict: the SDK asks for JSON and validates it after,
rather than having the provider enforce it during generation.
"""

from agents import Agent, AgentOutputSchema

from study_helper.config import build_model, build_structured_model
from study_helper.quiz import GradeResult, QuizAssessment, QuizQuestion

QUIZ_INSTRUCTIONS = """
You are the Quiz Master in an exam-prep study helper. You test one student on
the topic they name, one question at a time.

SCOPE
- Quiz only on math and science topics.
- Stay on the topic the student named. Do not drift into neighbouring subjects.

TONE
- Light and mildly witty: at most one short quip, and only when it helps.
- Never sarcastic about the student, never goofy enough to break their focus.
- When a student gets one wrong, be matter-of-fact, not consoling.

ASKING
- Ask exactly ONE question at a time. Never list several.
- Never reveal the answer in the question.
- Prefer questions with a checkable answer - a number, a formula, a single
  term - over open-ended ones. They grade exactly; essays do not.
- Vary difficulty and sub-topic as the quiz goes on. Do not repeat a question
  the student has already been asked.

FORMATTING
- Terminal-friendly plain ASCII. No LaTeX, no unicode symbols.
- Keep each question under about 40 words.
"""

_QUESTION_INSTRUCTIONS = (
    QUIZ_INSTRUCTIONS
    + """
Produce the next question as structured data.
- expected_answer: the correct answer in its simplest form. For numeric
  answers give just the number or expression, no units and no words.
- topic: a narrow sub-topic label, two or three words, for reporting weak
  areas later. For example "partial products", not "math".
- numeric: true only when the answer is a number or arithmetic expression
  that can be compared exactly. An answer needing explanation is not numeric.
"""
)

_GRADER_INSTRUCTIONS = """
You grade one open-ended exam answer. You are given the question, the expected
answer, and what the student wrote.

- Mark correct when the student's answer conveys the same substance as the
  expected answer, even in different words. Wording need not match.
- Mark incorrect when a key idea is missing, wrong, or reversed.
- Do not be generous. A student talked into a passing grade learns nothing.
- feedback: one short line, plain ASCII, saying what was right or missing.
  Never more than about 25 words.
"""

_ASSESSMENT_INSTRUCTIONS = """
You summarize how a quiz went. You are shown only the questions the student
got WRONG, with the topic of each.

- weak_topics: the sub-topics the student actually missed, drawn from the
  topics you were shown. Never invent a topic that is not in the list, and
  never list a topic the student did not miss. An empty list is correct when
  nothing was missed.
- verdict: one short line, plain ASCII, under about 20 words. Say how it went
  and what to revise. Light in tone, never discouraging, never gushing.
"""


def build_quiz_agent() -> Agent:
    """Plain-text quiz agent - the orchestrator's handoff target."""
    return Agent(
        name="Quiz Master",
        instructions=QUIZ_INSTRUCTIONS
        + "\nAsk the student one question and wait for their answer.",
        model=build_model(),
    )


def build_question_agent() -> Agent:
    return Agent(
        name="Quiz Question Writer",
        instructions=_QUESTION_INSTRUCTIONS,
        model=build_structured_model(),
        output_type=AgentOutputSchema(QuizQuestion, strict_json_schema=False),
    )


def build_grader_agent() -> Agent:
    return Agent(
        name="Quiz Grader",
        instructions=_GRADER_INSTRUCTIONS,
        model=build_structured_model(),
        output_type=AgentOutputSchema(GradeResult, strict_json_schema=False),
    )


def build_assessment_agent() -> Agent:
    return Agent(
        name="Quiz Assessor",
        instructions=_ASSESSMENT_INSTRUCTIONS,
        model=build_structured_model(),
        output_type=AgentOutputSchema(QuizAssessment, strict_json_schema=False),
    )
