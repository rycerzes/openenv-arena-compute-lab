# SPDX-License-Identifier: BSD-3-Clause

"""Office work: find the earliest meeting slot from constraints spread over an inbox.

Participants email their busy times in their own UTC offset. Higher difficulty
adds a correction email that changes an earlier constraint, a room that is only
free in some windows, and a participant who only takes morning meetings. The
answer is the earliest start on a 15-minute grid, in the organizer's time.
"""

import datetime as dt

from ..sandbox import Workspace
from .base import Episode, Family, Tool, ToolError, arg, file_tools, submit_action, submit_hint

PEOPLE = ["Ana", "Bram", "Chen", "Dalia", "Emeka", "Freya", "Goran", "Hana"]
DAY0 = dt.date(2026, 10, 13)  # a Tuesday; the conversion origin
# Windows start on a Mon/Tue/Wed between mid-October and late November, so dates vary per instance.
WINDOW_STARTS = [DAY0 + dt.timedelta(d) for d in range(-1, 45) if (DAY0 + dt.timedelta(d)).weekday() <= 2]
ORG_OFFSET = 1  # organizer's time is UTC+01:00
EPOCH = dt.datetime.combine(DAY0, dt.time(0))  # constraints are kept in UTC minutes since this


def _fmt(d: dt.datetime) -> str:
    return d.strftime("%a %d %b %Y %H:%M")


def _slot(local: dt.datetime, minutes: int) -> str:
    return f"- {_fmt(local)} to {(local + dt.timedelta(minutes=minutes)).strftime('%H:%M')}"


def _to_utc(local: dt.datetime, offset: int) -> int:
    """UTC minutes since EPOCH for a wall-clock time at UTC+offset hours."""
    return int((local - EPOCH).total_seconds() // 60) - offset * 60


def _to_local(minutes: int, offset: int) -> dt.datetime:
    return EPOCH + dt.timedelta(minutes=minutes + offset * 60)


def make(difficulty, rng):
    for _ in range(400):
        episode = _attempt(difficulty, rng)
        if episode:
            return episode
    raise RuntimeError("could not build a solvable inbox")


def _attempt(difficulty, rng):
    n_people = {1: 2, 2: 3, 3: 4, 4: 5}[difficulty]
    people = rng.sample(PEOPLE, n_people)
    duration = rng.choice([30, 45, 60, 90])
    day0 = rng.choice(WINDOW_STARTS)
    days = [day0 + dt.timedelta(d) for d in range(3)]
    offsets = {p: (ORG_OFFSET if difficulty < 3 else rng.choice([-5, 0, 1, 2, 5])) for p in people}

    busy = {p: [] for p in people}  # (start, end) in UTC minutes since EPOCH
    emails = []
    for p in people:
        off = offsets[p]
        lines = []
        for day in days:
            # an early block on the first day keeps "first slot of the window" from being the answer
            starts = [(9, 0)] if day == days[0] and rng.random() < 0.6 else []
            starts += [(rng.randint(9, 15), rng.choice([0, 15, 30, 45])) for _ in range(rng.randint(1, 3))]
            for h, m in starts:
                length = rng.choice([30, 45, 60, 90, 120])
                local = dt.datetime.combine(day, dt.time(h, m))
                start = _to_utc(local, off)
                busy[p].append((start, start + length))
                lines.append(_slot(local, length))
        tz = f"UTC{off:+03d}:00"
        emails.append((p, f"My availability ({tz})", f"Hi,\nAll times below are my local time ({tz}). I work 09:00-17:00 local. Busy:\n" + "\n".join(lines) + "\nThanks,\n" + p))

    morning_only = None
    if difficulty == 4:
        morning_only = rng.choice(people)
        emails.append((morning_only, "One more thing", f"Hi,\nI can only join meetings that end by 12:00 my local time.\n{morning_only}"))
    if difficulty >= 2:
        p = rng.choice(people)
        old = rng.randrange(len(busy[p]))
        start, end = busy[p][old]
        shift = rng.choice([-60, 60, 120])
        busy[p][old] = (start + shift, end + shift)
        old_local = _to_local(start, offsets[p])
        new_local = old_local + dt.timedelta(minutes=shift)
        emails.append((p, "Correction to my availability", f"Hi,\nCorrection: my {_fmt(old_local)} block moved to {_fmt(new_local)} (same length, my local time).\n{p}"))
    room = None
    if difficulty >= 3:
        room = []
        lines = []
        for day in days:
            h = rng.randint(9, 13)
            length = rng.choice([180, 240, 300])
            local = dt.datetime.combine(day, dt.time(h, 0))
            start = _to_utc(local, ORG_OFFSET)
            room.append((start, start + length))
            lines.append(_slot(local, length))
        emails.append(("Facilities", "Room B2 availability", "Room B2 is free only in these windows (UTC+01:00):\n" + "\n".join(lines)))

    def ok(t, with_busy=True):
        end = t + duration
        for p in people:
            off = offsets[p] * 60
            local_start, local_end = t + off, end + off
            day_start = (local_start // 1440) * 1440
            if local_start < day_start + 9 * 60 or local_end > day_start + 17 * 60:
                return False
            if morning_only == p and local_end > day_start + 12 * 60:
                return False
            if with_busy and any(s < end and t < e for s, e in busy[p]):
                return False
        if room is not None and not any(s <= t and end <= e for s, e in room):
            return False
        return True

    first = _to_utc(dt.datetime.combine(day0, dt.time(0)), ORG_OFFSET)  # 00:00 on the first day, organizer's time
    candidates = [t for t in range(first, first + 3 * 1440, 15) if ok(t)]
    if not candidates or candidates[0] < first + 60:
        return None
    # The inbox must matter: reject instances whose answer is just the first slot that working
    # hours and the room allow (otherwise one constant answer solves most instances).
    if candidates[0] == next(t for t in range(first, first + 3 * 1440, 15) if ok(t, with_busy=False)):
        return None
    expected = _to_local(candidates[0], ORG_OFFSET).strftime("%Y-%m-%d %H:%M")

    rng.shuffle(emails)
    inbox = [{"id": i + 1, "from": f, "subject": s, "body": b} for i, (f, s, b) in enumerate(emails)]
    brief = f"Book a {duration}-minute meeting with {', '.join(people)} between {days[0]:%a %d %b %Y} and {days[-1]:%a %d %b %Y}."
    if room:
        brief += " It must be held in room B2."
    files = {"brief.txt": brief + " Use the earliest start that works for everyone, on a 15-minute grid.\n"}
    ws = Workspace(files)

    def list_emails(args):
        return "\n".join(f"{m['id']}: from {m['from']} | {m['subject']}" for m in inbox)

    def read_email(args):
        i = arg(args, "id", int)
        if not 1 <= i <= len(inbox):
            raise ToolError(f"no email {i}; ids 1-{len(inbox)}")
        m = inbox[i - 1]
        return f"From: {m['from']}\nSubject: {m['subject']}\n\n{m['body']}"

    def grade(answer):
        got = answer.get("start") if isinstance(answer, dict) else None
        if not isinstance(got, str):
            return 0.0, 'answer needs "start" as "YYYY-MM-DD HH:MM"'
        return (1.0, "correct") if got.strip() == expected else (0.0, "not the earliest valid start")

    tools = file_tools(ws, files) + [
        Tool("list_emails", '{"tool": "list_emails", "args": {}}  list the inbox', list_emails),
        Tool("read_email", '{"tool": "read_email", "args": {"id": 1}}  show one email', read_email),
    ]
    return Episode(
        question="Schedule the meeting described in brief.txt using the constraints in the inbox. Answer in the organizer's time (UTC+01:00).",
        tools=tools,
        answer_hint=submit_hint({"start": "YYYY-MM-DD HH:MM"}),
        grade=grade,
        workspace=ws,
        oracle=[submit_action({"start": expected})],
    )


FAMILY = Family("inbox_scheduling", "office_white_collar", "Earliest meeting slot from constraints spread across emails, offsets and corrections", make)
