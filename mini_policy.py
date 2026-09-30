"""What mini-swe-agent's model may run: `echo …` or one `python3 recruiter_cli.py` call.

mini hands the model a "bash" tool.  mini_env.RecruiterEnvironment runs only
what parse() accepts, as an argv list without a shell, so the model can't
chain commands, set variables, redirect output, read files or pick real vs
mock.  Stdlib only: grounding.py and compare_agents.py classify commands with
it without importing minisweagent.
"""
import shlex

SUBCOMMANDS = ("find-talents", "get-match-id-from-profile-id", "generate-jd",
               "score-candidates", "candidate-insights")
# They trigger calculations on VIRA (recal_briq), so real mode asks a person first.
SIDE_EFFECTS = frozenset({"score-candidates", "candidate-insights"})
MODES = ("real", "mock")
MAX_COMMAND = 4096
_OPERATOR_CHARS = set(";&|<>()")
ONLY = "only `python3 recruiter_cli.py <subcommand> <flags>` and `echo <text>` can run"


def _tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def parse(command: str, mode: str | None = None) -> tuple[str, str | list[str]]:
    """-> ("echo", text) | ("cli", [subcommand, *flags]) | ("refused", reason).

    `mode` is the host's mode: a `--mode` the model writes must match it.  None
    accepts either mode (classifying a stored trace).  The returned flags never
    contain --mode; the environment adds the host's.
    """
    if not isinstance(command, str) or not command.strip():
        return "refused", "empty command"
    if len(command) > MAX_COMMAND:
        return "refused", f"command longer than {MAX_COMMAND} characters"
    if any(c in command for c in "\n\r\0"):
        return "refused", "one command per call, on one line"
    try:
        tokens = _tokens(command)
    except ValueError:
        return "refused", "unbalanced quotes"
    if any(t and set(t) <= _OPERATOR_CHARS for t in tokens):
        return "refused", "this is not a shell: no pipes, redirection, `;`, `&&` or subshells " \
                          "(quote text that contains | or ;)"
    if tokens[0] == "echo":
        return "echo", " ".join(tokens[1:])
    if tokens[:2] != ["python3", "recruiter_cli.py"]:
        return "refused", ONLY
    rest = tokens[2:]
    if rest and (rest[0] == "--mode" or rest[0].startswith("--mode=")):
        if rest[0] == "--mode":
            given, rest = (rest[1] if len(rest) > 1 else ""), rest[2:]
        else:
            given, rest = rest[0].split("=", 1)[1], rest[1:]
        if given not in MODES:
            return "refused", f"unknown mode {given!r}"
        if mode is not None and given != mode:
            return "refused", f"the mode is set by the host ({mode}); leave --mode out"
    if not rest or rest[0] not in SUBCOMMANDS:
        return "refused", f"the subcommand must follow recruiter_cli.py: one of {', '.join(SUBCOMMANDS)}"
    if any(t == "--confirmed" or t.startswith("--confirmed=") for t in rest):
        return "refused", "confirmation comes from a person, not from a flag"
    return "cli", rest
