from academic_assistant.utils.prompts import pdf_summary_prompt, summaries_qa_prompt

SECTIONS = ["## Key Concepts", "## Detailed Notes", "## Action Items / Study Questions"]


def test_pdf_prompt_includes_subject_context_and_sections():
    text = pdf_summary_prompt("numerical methods").format_messages(context="CTX-123")[0].content
    assert "numerical methods" in text
    assert "CTX-123" in text
    for section in SECTIONS:
        assert section in text


def test_qa_prompt_includes_subject_question_and_context():
    text = summaries_qa_prompt("graph theory").format_messages(context="CTX", question="What is a cut?")[0].content
    assert "expert academic tutor for graph theory" in text
    assert "What is a cut?" in text
    assert "CTX" in text


def test_subject_with_braces_is_not_treated_as_template():
    text = pdf_summary_prompt("sets {A, B}").format_messages(context="x")[0].content
    assert "sets {A, B}" in text


def test_math_rules_survive_templating():
    text = pdf_summary_prompt("x").format_messages(context="")[0].content
    assert r"Do NOT use LaTeX \( \), \[ \]" in text
    assert r"\begin{equation}" in text
