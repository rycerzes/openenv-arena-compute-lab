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
MAX_OUTPUT_CHARS = 1500


def _limit_child() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_S, CPU_LIMIT_S))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        if sys.platform.startswith("linux"):
            resource.setrlimit(
                resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES)
            )
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ImportError, ValueError, OSError):
        pass
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(NOBODY_GID)
        os.setuid(NOBODY_UID)


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
            with open(os.path.join(self.path, name), "w", encoding="utf-8") as f:
                f.write(content)
        if os.geteuid() == 0:
            for name in [*files, ""]:
                os.chown(os.path.join(self.path, name), NOBODY_UID, NOBODY_GID)
        os.chmod(self.path, 0o700)

    def run_python(self, code: str) -> str:
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": self.path,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        proc = subprocess.Popen(
            [sys.executable, "-I", "-"],
            cwd=self.path,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=_limit_child,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(code.encode("utf-8"), timeout=WALL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return clip(out.decode("utf-8", "replace")) + "\n[killed: time limit]"
        text = out.decode("utf-8", "replace")
        if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
            text += "\n[killed: CPU time limit]"
        if not text.strip():
            text = "(no output; use print() to show results)"
        return clip(text)

    def close(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
