from academic_assistant.utils.math_format import normalize_math_delimiters


def test_inline_math_becomes_dollars():
    assert normalize_math_delimiters(r"Let \( Ax = b \) hold.") == "Let $Ax = b$ hold."


def test_display_math_becomes_double_dollars_on_own_lines():
    assert normalize_math_delimiters(r"\[ \|x\|_2 \]") == "$$\n\\|x\\|_2\n$$"


def test_multiline_display_math():
    text = "\\[\nA = U \\Sigma V^T\n\\]"
    assert normalize_math_delimiters(text) == "$$\nA = U \\Sigma V^T\n$$"


def test_mixed_and_existing_dollars_untouched():
    text = r"Already $x$ and \(y\) and \[z\]"
    assert normalize_math_delimiters(text) == "Already $x$ and $y$ and $$\nz\n$$"


def test_plain_text_unchanged():
    assert normalize_math_delimiters("No math (just parentheses) [and brackets].") == (
        "No math (just parentheses) [and brackets]."
    )
