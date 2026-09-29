"""The only "bash" mini-swe-agent's model gets: recruiter_cli and echo, no shell.

minisweagent's LocalEnvironment runs whatever the model writes through
`subprocess(shell=True)` with this process's whole environment, so one injected
`env` or `cat .env` would put every credential into model context.  This
environment runs only what mini_policy.parse() accepts:

  * `echo …` is answered here; no process starts.
  * `python3 recruiter_cli.py <subcommand> …` runs as an argv list, without a
    shell, with this interpreter, and with the host's --mode.
  * anything else comes back as a refusal (returncode 126); nothing runs.

The child gets a minimal environment: no provider keys, nothing from the
user's shell.  VIRA credentials arrive as `secrets`, kept off the config so
they are never serialised or offered to the prompt templates.
"""
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Literal

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.utils.serialize import recursive_merge

import mini_policy

HERE = Path(__file__).resolve().parent
CLI = HERE / "recruiter_cli.py"
PASS_THROUGH = ("PYTHON_DOTENV_DISABLED", "EVENTS_LOG", "TZ")


class RecruiterEnvironmentConfig(LocalEnvironmentConfig):
    mode: Literal["real", "mock"]
    cwd: str = str(HERE)
    timeout: int = 45            # above recruiter_cli's 30 s request timeout
    max_output: int = 20000


def _refused(reason: str) -> dict[str, Any]:
    return {"output": f"Refused: {reason}.\n", "returncode": 126, "exception_info": "",
            "extra": {"refused": True}}


class RecruiterEnvironment(LocalEnvironment):
    def __init__(self, *, secrets: dict[str, str] | None = None,
                 approve: Callable[[list[str]], bool] | None = None, **kwargs):
        """`approve(argv)` is asked before a side-effecting call in real mode; None refuses it."""
        super().__init__(config_class=RecruiterEnvironmentConfig, **kwargs)
        self._secrets = dict(secrets or {})
        self._approve = approve

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        kind, value = mini_policy.parse(action.get("command", ""), self.config.mode)
        if kind == "refused":
            return _refused(value)
        if kind == "echo":
            output = {"output": f"{value}\n", "returncode": 0, "exception_info": ""}
            self._check_finished(output)          # echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
            return output
        if self.config.mode == "real" and value[0] in mini_policy.SIDE_EFFECTS:
            if self._approve is None or not self._approve(list(value)):
                return _refused(f"the user did not approve {value[0]}")
        # Never _check_finished here: VIRA's reply must not be able to end the run.
        return self._run_cli(value, cwd, timeout or self.config.timeout)

    def _child_env(self) -> dict[str, str]:
        env = {"PATH": os.defpath, "LANG": "C.UTF-8"}
        env |= {k: os.environ[k] for k in PASS_THROUGH if k in os.environ}
        return env | self._secrets | self.config.env

    def _run_cli(self, args: list[str], cwd: str, timeout: int) -> dict[str, Any]:
        argv = [sys.executable, "-E", "-s", str(CLI), "--mode", self.config.mode, *args]
        proc = subprocess.Popen(argv, cwd=cwd or self.config.cwd, env=self._child_env(),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", start_new_session=os.name == "posix")
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL) if os.name == "posix" else proc.kill()
            proc.communicate()
            return {"output": "", "returncode": -1,
                    "exception_info": f"recruiter_cli timed out after {timeout}s",
                    "extra": {"exception_type": "TimeoutExpired"}}
        if proc.returncode != 0:
            # argparse's own error line helps the model fix its flags; a traceback never
            # reaches it (it can name hosts and paths).
            usage = [line for line in err.splitlines() if ": error: " in line]
            out += (usage[-1] if usage else f"recruiter_cli failed (exit {proc.returncode})") + "\n"
        if len(out) > self.config.max_output:
            out = out[:self.config.max_output] + "\n[output truncated]\n"
        return {"output": out, "returncode": proc.returncode, "exception_info": ""}

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        # LocalEnvironment adds os.environ here, which would put every key in reach of
        # the prompt templates.
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)
