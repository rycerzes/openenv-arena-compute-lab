# SPDX-License-Identifier: BSD-3-Clause

"""Media production: retime an SRT subtitle file from the notes.

Difficulty 1-2: a linear correction from two sync points. Difficulty 3: a frame
rate conversion plus one sync point. Difficulty 4: like 3, and a scene inserted
in the release shifts every cue that starts after a given original time.
Cues are named by their text; times are graded to within 2 ms.
"""

import re
from collections import Counter

from ..sandbox import Workspace
from .base import Episode, Family, file_tools, submit_action, submit_hint

LINES = [
    "Get down!", "Where were you last night?", "It's not what it looks like.", "We need to leave. Now.",
    "Did anyone see the car?", "Keep the engine running.", "That's the third time this week.",
    "I told you, I don't know him.", "Turn left at the bridge.", "Is this seat taken?", "The tide is coming in.",
    "Who signed for the package?", "Call me when you land.", "Don't touch that switch.", "We're out of coffee again.",
    "Check the north gate.", "How long has it been broken?", "She left before sunrise.", "Bring the blue folder.",
    "Nobody leaves this room.", "Wait for my signal.", "That's not my handwriting.", "The train was late.",
    "Can you hear the bells?", "Lock the back door.", "He paid in cash.", "It's colder than yesterday.",
]

RULES = """# Retiming rules
- Work in milliseconds. Round each final time to the nearest millisecond (half up).
- A linear correction maps every original time t to a * t + b; find a and b from the notes.
- A frame-rate conversion from F1 fps to F2 fps scales every original time by F1 / F2.
- Scene inserts add their length to every cue that starts after the given original time
  (both the start and end of that cue move).
- Answer times as HH:MM:SS,mmm.
"""


def _ts(ms: int) -> str:
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_ts(text: str):
    m = re.fullmatch(r"\s*(\d+):(\d\d):(\d\d)[,.](\d{3})\s*", text)
    if not m:
        return None
    h, mi, s, ms = map(int, m.groups())
    return ((h * 60 + mi) * 60 + s) * 1000 + ms


def _round(x: float) -> int:
    return int(x + 0.5)


def make(difficulty, rng):
    n = {1: 12, 2: 24, 3: 40, 4: 60}[difficulty]
    texts = rng.sample(LINES, min(n, len(LINES))) + [f"[{rng.choice(['music', 'door slams', 'laughter', 'phone rings'])}]" for _ in range(max(0, n - len(LINES)))]
    rng.shuffle(texts)
    cues, t = [], rng.randint(10000, 30000)  # leaves room for negative offsets
    for text in texts:
        t += rng.randint(1500, 30000)
        cues.append((t, t + rng.randint(1200, 4500), text))
    counts = Counter(text for _, _, text in cues)
    unique = [i for i, (_, _, text) in enumerate(cues) if counts[text] == 1]  # cues a note can name
    notes = []
    if difficulty <= 2:
        a = 1 + rng.choice([-1, 1]) * rng.uniform(0.0005, 0.004) if difficulty == 2 else 1.0
        b = rng.randint(-8000, 8000)
        i, j = sorted(rng.sample(unique, 2))
        if j - i < n // 3:
            i, j = unique[0], unique[-1]
        si, sj = _round(a * cues[i][0] + b), _round(a * cues[j][0] + b)
        notes.append(f'- The cue "{cues[i][2]}" should start at {_ts(si)}.')
        notes.append(f'- The cue "{cues[j][2]}" should start at {_ts(sj)}.')
        # the agent derives a and b from the two (rounded) sync points
        a = (sj - si) / (cues[j][0] - cues[i][0])
        b = si - a * cues[i][0]

        def mapping(x, start):
            return a * x + b

    else:
        f1, f2 = rng.choice([(25, 23.976), (23.976, 25), (24, 25), (25, 24)])
        k = rng.choice(unique)
        sk = _round(cues[k][0] * f1 / f2) + rng.randint(-5000, 5000)
        off = sk - cues[k][0] * f1 / f2
        cut, extra = None, 0
        if difficulty == 4:
            cut = cues[rng.randrange(n // 3, n - 3)][0] - rng.randint(100, 1000)
            extra = rng.randint(2000, 15000)
        sync_after_insert = cut is not None and cues[k][0] > cut
        if sync_after_insert:
            off -= extra
        steps = "that conversion and the scene insert" if sync_after_insert else "that conversion"
        notes.append(f"- The subtitles were timed for {f1} fps; the release runs at {f2} fps.")
        notes.append(f'- After {steps}, the cue "{cues[k][2]}" should start at {_ts(sk)}.')
        if cut is not None:
            notes.append(f"- The release adds a {extra} ms scene; it affects cues that start after original time {_ts(cut)}.")

        def mapping(x, start):
            return x * f1 / f2 + off + (extra if cut is not None and start > cut else 0)

    def final(idx, which):
        start, end, _ = cues[idx]
        return _round(mapping(start if which == "start" else end, start))

    asks = rng.sample(unique, 3)
    expected = {}
    for q, idx in enumerate(asks):
        which = "start" if q % 2 == 0 else "end"
        expected[f"{which}_{q + 1}"] = (final(idx, which), f'{which} of the cue "{cues[idx][2]}"')
    srt = "\n".join(f"{i}\n{_ts(s)} --> {_ts(e)}\n{text}\n" for i, (s, e, text) in enumerate(cues, 1))
    files = {"movie.srt": srt, "notes.md": "# Sync notes\n" + "\n".join(notes) + "\n", "rules.md": RULES}
    ws = Workspace(files)

    def grade(answer):
        if not isinstance(answer, dict):
            return 0.0, "answer must be an object"
        right = 0
        for key, (ms, _) in expected.items():
            got = _parse_ts(str(answer.get(key, "")))
            right += got is not None and abs(got - ms) <= 2
        return float(right == len(expected)), f"{right}/{len(expected)} times correct"

    hint = "Answer fields:\n" + "\n".join(f"- {k}: {d}, as HH:MM:SS,mmm" for k, (_, d) in expected.items())
    hint += "\n" + submit_hint({k: "..." for k in expected})
    return Episode(
        question="Retime movie.srt for the release using notes.md and rules.md.",
        tools=file_tools(ws, files),
        answer_hint=hint,
        grade=grade,
        workspace=ws,
        oracle=[submit_action({k: _ts(ms) for k, (ms, _) in expected.items()})],
    )


FAMILY = Family("subtitle_sync", "media_content_production", "Retime subtitles from sync points, frame-rate conversion and scene inserts", make)
