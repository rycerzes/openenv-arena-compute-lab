# SPDX-License-Identifier: BSD-3-Clause

"""Cybersecurity: find the brute-force login that succeeded, using grep over sshd logs.

No python tool: the agent searches with grep (regex, run in the sandbox) and paged
reads. Higher difficulty adds a decoy brute-force source that never succeeds, an
attacker rotating through addresses in one /24, and log rotation (the first
success may sit in the older auth.log.1).
"""

import datetime as dt
import json

from ..sandbox import Workspace
from ..tasks.base import AnswerField
from .base import Episode, Family, Tool, ToolError, arg, fields_grader, fields_hint, fields_oracle, file_tools

USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy"]
PROBES = ["admin", "root", "test", "oracle", "ubuntu", "postgres", "git", "pi"]
GREP = r'''
import json, re, sys
args = json.loads(ARGS)
try:
    pat = re.compile(args["pattern"])
except re.error as e:
    print(f"bad regex: {e}"); raise SystemExit
total, shown = 0, []
with open(args["path"], encoding="utf-8") as f:
    for n, line in enumerate(f, 1):
        if pat.search(line):
            total += 1
            if len(shown) < args["max"]:
                shown.append(f"{n}: {line.rstrip()[:200]}")
print("\n".join(shown))
print(f"[{total} matching lines" + (f"; showing first {len(shown)}]" if total > len(shown) else "]"))
'''


def _line(t, host, pid, msg):
    return f"{t:%b %d %H:%M:%S} {host} sshd[{pid}]: {msg}"


def make(difficulty, rng):
    size = {1: 300, 2: 900, 3: 2500, 4: 5000}[difficulty]
    host = rng.choice(["web01", "bastion", "app-2", "db-gw"])
    start = dt.datetime(2026, 10, rng.randint(5, 20), 0, 0, 0)
    span = 3 * 24 * 3600
    office = [f"10.20.{rng.randint(0, 9)}.{rng.randint(2, 250)}" for _ in range(8)]
    events = []

    def at(sec):
        return start + dt.timedelta(seconds=sec)

    for _ in range(size):
        t = rng.randrange(span)
        user, ip, port, pid = rng.choice(USERS), rng.choice(office), rng.randint(40000, 65000), rng.randint(1000, 30000)
        if rng.random() < 0.12:
            events.append((t, _line(at(t), host, pid, f"Failed password for {user} from {ip} port {port} ssh2")))
        events.append((t + 1, _line(at(t + 1), host, pid, f"Accepted publickey for {user} from {ip} port {port} ssh2")))

    net = f"{rng.randint(100, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}"
    attacker = f"{net}.{rng.randint(2, 250)}"
    ips = [attacker] if difficulty < 3 else [f"{net}.{rng.randint(2, 250)}" for _ in range(4)] + [attacker]
    t0 = rng.randrange(span // 6, span // 2)
    t = t0
    for i in range(rng.randint(40, 120)):
        if len(ips) == 1:
            ip = attacker
        else:  # two in three attempts come from the other addresses in the /24
            ip = rng.choice(ips[:-1] if i % 3 else ips)
        user = rng.choice(PROBES + USERS[:3])
        msg = f"Failed password for {'invalid user ' if user in PROBES else ''}{user} from {ip} port {rng.randint(30000, 65000)} ssh2"
        events.append((t, _line(at(t), host, rng.randint(1000, 30000), msg)))
        t += rng.randint(1, 9)
    victim = rng.choice(USERS)
    t_success = t + rng.randint(5, 40)
    events.append((t_success, _line(at(t_success), host, 31337, f"Accepted password for {victim} from {attacker} port {rng.randint(30000, 65000)} ssh2")))
    later = t_success + rng.randint(600, 7200)
    other = rng.choice([u for u in USERS if u != victim])
    events.append((later, _line(at(later), host, 31338, f"Accepted password for {other} from {attacker} port {rng.randint(30000, 65000)} ssh2")))
    events.append((t_success + 30, f"{at(t_success + 30):%b %d %H:%M:%S} {host} sudo:   {victim} : TTY=pts/3 ; PWD=/home/{victim} ; USER=root ; COMMAND=/bin/cat /etc/shadow"))
    if difficulty >= 2:
        # office users who log in with passwords; one always does so before the attack succeeds
        times = [rng.randrange(0, t_success)] + [rng.randrange(span) for _ in range(rng.randint(4, 12))]
        for tp in times:
            events.append((tp, _line(at(tp), host, rng.randint(1000, 30000), f"Accepted password for {rng.choice(USERS)} from {rng.choice(office)} port {rng.randint(40000, 65000)} ssh2")))
        # a decoy source that brute-forces invalid users and never gets in
        decoy = f"{rng.randint(100, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(2, 250)}"
        td = rng.randrange(span)
        for _ in range(rng.randint(150, 300)):
            user = rng.choice(PROBES)
            events.append((td, _line(at(td), host, rng.randint(1000, 30000), f"Failed password for invalid user {user} from {decoy} port {rng.randint(30000, 65000)} ssh2")))
            td += rng.randint(1, 4)

    events.sort()
    lines = [e[1] for e in events]
    files = {}
    if difficulty == 4:
        cut = next(i for i, e in enumerate(events) if e[0] > t_success) + rng.randint(5, 50)
        files["auth.log.1"] = "\n".join(lines[:cut]) + "\n"
        files["auth.log"] = "\n".join(lines[cut:]) + "\n"
    else:
        files["auth.log"] = "\n".join(lines) + "\n"
    ws = Workspace(files)

    def grep(args):
        pattern = arg(args, "pattern")
        path = arg(args, "path").strip()
        if path not in files:
            raise ToolError(f"no file {path!r}; files: {', '.join(files)}")
        if len(pattern) > 300:
            raise ToolError("pattern too long")
        payload = json.dumps({"pattern": pattern, "path": path, "max": 20})
        return ws.run_python(f"ARGS = {payload!r}\n{GREP}", limit=4500)  # already bounded to 20 lines

    fields = [
        AnswerField("attacker_ip", attacker, "text", "IP address of that first successful login", choices=[attacker]),
        AnswerField("user", victim, "text", "account compromised by that login", choices=[victim]),
        AnswerField("time", f"{at(t_success):%b %d %H:%M:%S}", "text", 'timestamp of that login as written in the log, e.g. "Oct 07 03:14:07"'),
    ]
    note = ""
    if difficulty >= 3:
        note += " The attacker rotated addresses within one /24 network: every address in that network counts as the attacking source."
    if difficulty == 4:
        note += " Logs rotate: auth.log.1 is older than auth.log."
    return Episode(
        question=(
            f"{host} was brute-forced over SSH and an attacker got in. The attacking source is the address(es) that "
            f"failed logins for invalid user names. Find the first successful login from the attacking source.{note}"
        ),
        tools=file_tools(ws, files, python=False) + [Tool("grep", '{"tool": "grep", "args": {"pattern": "<Python regex>", "path": "auth.log"}}  matching lines with line numbers (first 20) and the total count', grep)],
        answer_hint=fields_hint(fields),
        grade=fields_grader(fields),
        workspace=ws,
        oracle=fields_oracle(fields),
    )


FAMILY = Family("auth_forensics", "cybersecurity", "Find the brute-force login that succeeded with grep over sshd logs, decoys and rotation", make)
