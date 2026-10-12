# SPDX-License-Identifier: BSD-3-Clause

"""Run model-written Python in a per-episode workspace with resource limits.

The expected answers never touch the workspace: they live only in the server
process. When the server runs as root (as in the container), the child drops
to the unprivileged `nobody` user, so it cannot read the server's memory or
the environment source.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile

NOBODY_UID = 65534
NOBODY_GID = 65534
WALL_TIMEOUT_S = 15
CPU_LIMIT_S = 10
MEMORY_LIMIT_BYTES = 1 << 30
MAX_OUTPUT_CHARS = 1000


# Runs inside the child before the agent's code: resource limits are set by the
# (already unprivileged) child itself, so no Python runs between fork and exec.
PRELUDE = f"""
import resource, sys
def _cap(name, value):
    try:
        resource.setrlimit(getattr(resource, name), (value, value))
    except (AttributeError, ValueError, OSError):
        pass
_cap("RLIMIT_CPU", {CPU_LIMIT_S})
_cap("RLIMIT_FSIZE", {16 << 20})
if sys.platform.startswith("linux"):
    _cap("RLIMIT_AS", {MEMORY_LIMIT_BYTES})
    _cap("RLIMIT_NPROC", 64)
del _cap, resource
_code = sys.stdin.read()
exec(compile(_code, "<tool>", "exec"), {{"__name__": "__main__"}})
"""


def clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the head and tail of long output."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} chars cut] ...\n{text[-tail:]}"


class Workspace:
    """A temporary folder holding one episode's task files."""

    def __init__(self, files: dict[str, str]):
        self.path = tempfile.mkdtemp(prefix="compute-lab-")
        for name, content in files.items():
            self.write(name, content)
        if os.geteuid() == 0:
            os.chown(self.path, NOBODY_UID, NOBODY_GID)
        os.chmod(self.path, 0o700)

    def write(self, name: str, content: str) -> None:
        """Create or replace a file, owned by the sandbox user when running as root."""
        path = os.path.join(self.path, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if os.geteuid() == 0:
            os.chown(path, NOBODY_UID, NOBODY_GID)

    def run_python(self, code: str, limit: int = MAX_OUTPUT_CHARS) -> str:
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": self.path,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        # No preexec_fn: it can deadlock between fork and exec when the server has threads.
        drop = {"user": NOBODY_UID, "group": NOBODY_GID, "extra_groups": []} if os.geteuid() == 0 else {}
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", PRELUDE],
            cwd=self.path,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            **drop,
        )
        try:
            out, _ = proc.communicate(code.encode("utf-8"), timeout=WALL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return clip(out.decode("utf-8", "replace"), limit) + "\n[killed: time limit]"
        text = out.decode("utf-8", "replace")
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            text += "\n[killed: CPU time limit]"
        if not text.strip():
            text = "(no output; use print() to show results)"
        return clip(text, limit)

    def close(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
