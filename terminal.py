"""Text written by a model or by VIRA, made safe to print to your terminal.

Printed raw, an ESC sequence can clear the screen, retitle the window, write the
clipboard (OSC 52) or hide text; a bidi override can make an approval prompt
read differently from what will run.  Stdlib only: run_mini uses it without
importing agent_kit.
"""
import re

# C0 controls except tab and newline, DEL, C1 controls, zero-width and bidi controls.
_UNSAFE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]")


def printable(text) -> str:
    return _UNSAFE.sub("", str(text))
