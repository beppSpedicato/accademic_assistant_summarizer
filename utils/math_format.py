import re


_MATH_INLINE_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_MATH_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def normalize_math_delimiters(text: str) -> str:
    r"""Convert LaTeX \( ... \) and \[ ... \] into $...$ and $$...$$."""
    text = _MATH_DISPLAY_RE.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", text)
    text = _MATH_INLINE_RE.sub(lambda m: f"${m.group(1).strip()}$", text)
    return text
