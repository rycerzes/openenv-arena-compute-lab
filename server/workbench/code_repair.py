# SPDX-License-Identifier: BSD-3-Clause

"""Software engineering: repair seeded bugs in a small Python module.

The agent edits `lib.py` with exact-match replacements and runs the visible
tests. Grading runs hidden tests in a fresh sandbox: the reward is the share of
buggy functions whose hidden tests all pass, and 0 if a function that was
correct got broken. At difficulty 3 one bug, and at difficulty 4 two bugs, have no
failing visible test, so the agent has to check the code against SPEC.md.
"""

import datetime as dt
import json
import re

from ..sandbox import Workspace
from .base import Episode, Family, Tool, ToolError, arg, file_tools, submit_action

# Test-case argument generators used by FUNCTIONS below.


def _luhn_case(r):
    """A digit string (sometimes grouped by dashes) whose check digit is right 60% of the time."""
    body = [r.randint(0, 9) for _ in range(r.randint(1, 15))]
    total = 0
    for i, d in enumerate(reversed(body)):
        if i % 2 == 0:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    check = (10 - total % 10) % 10
    digits = body + [check if r.random() < 0.6 else (check + r.randint(1, 9)) % 10]
    text = "".join(map(str, digits))
    return "-".join(text[i : i + 4] for i in range(0, len(text), 4)) if r.random() < 0.4 else text


def _duration_case(r):
    if r.random() < 0.3:  # units in any order, upper case
        units = r.sample(["h", "m", "s"], r.randint(1, 3))
        return ("".join(f"{r.randint(1, 59)}{u}" for u in units).upper(),)
    units = sorted(r.sample(["h", "m", "s"], r.randint(1, 3)), key="hms".index)
    return ("".join(f"{r.randint(1, 59)}{u}" for u in units),)


def _roman(n):
    out = ""
    for value, sym in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= value:
            out, n = out + sym, n - value
    return out


def _business_case(r):
    start = dt.date(2026, 1, 1) + dt.timedelta(r.randint(0, 300))
    end = start + dt.timedelta(r.randint(0, 25))
    holidays = [(start + dt.timedelta(r.randint(0, 25))).isoformat() for _ in range(r.randint(0, 3))]
    return (start.isoformat(), end.isoformat(), holidays)


# name -> (source, spec line, bugs [(old, new)], case generator)
FUNCTIONS = {
    "clamp_mean": (
        '''def clamp_mean(xs, lo, hi):
    if not xs:
        return 0.0
    total = 0.0
    for x in xs:
        total += min(max(x, lo), hi)
    return total / len(xs)
''',
        "clamp_mean(xs, lo, hi): mean of the values after clipping each to [lo, hi]; 0.0 for an empty list.",
        [("min(max(x, lo), hi)", "max(min(x, lo), hi)"), ("return total / len(xs)", "return total / (len(xs) - 1)"), ("        return 0.0\n", "        return None\n")],
        lambda r: ([r.randint(-20, 20) for _ in range(r.randint(0, 6))], r.randint(-10, 0), r.randint(1, 10)),
    ),
    "parse_duration": (
        '''def parse_duration(text):
    units = {"h": 3600, "m": 60, "s": 1}
    total, number = 0, ""
    for ch in text.strip().lower():
        if ch.isdigit():
            number += ch
        elif ch in units and number:
            total += int(number) * units[ch]
            number = ""
        else:
            raise ValueError(f"bad duration: {text!r}")
    if number:
        raise ValueError(f"bad duration: {text!r}")
    return total
''',
        'parse_duration(text): seconds in a duration like "1h30m15s", "45m" or "2H"; units h, m, s in any case; ValueError otherwise.',
        [('"m": 60', '"m": 600'), ("text.strip().lower()", "text.strip()"), ("            number = \"\"\n        else:", "        else:")],
        _duration_case,
    ),
    "merge_intervals": (
        '''def merge_intervals(pairs):
    merged = []
    for start, end in sorted(pairs):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(p) for p in merged]
''',
        "merge_intervals(pairs): merge overlapping or touching (start, end) intervals; sorted list of tuples.",
        [("start <= merged[-1][1]", "start < merged[-1][1]"), ("max(merged[-1][1], end)", "end"), ("sorted(pairs)", "pairs")],
        lambda r: ([(s, s + r.randint(0, 6)) for s in (r.randint(0, 30) for _ in range(r.randint(2, 7)))],),
    ),
    "luhn_valid": (
        '''def luhn_valid(number):
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 2:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
''',
        "luhn_valid(number): Luhn checksum of the digits in the string (spaces and dashes ignored); fewer than 2 digits is invalid.",
        [("i % 2 == 1", "i % 2 == 0"), ("d -= 9", "d -= 10"), ("len(digits) < 2", "len(digits) < 1")],
        lambda r: (_luhn_case(r),),
    ),
    "roman_to_int": (
        '''def roman_to_int(text):
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(text):
        value = values[ch]
        if i + 1 < len(text) and values[text[i + 1]] > value:
            total -= value
        else:
            total += value
    return total
''',
        "roman_to_int(text): value of an upper-case Roman numeral such as MCMXCIV.",
        [("values[text[i + 1]] > value", "values[text[i + 1]] >= value"), ('"D": 500', '"D": 50'), ("            total -= value", "            total += value")],
        lambda r: (_roman(r.randint(1, 3999)),),
    ),
    "chunk": (
        '''def chunk(items, size):
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]
''',
        "chunk(items, size): consecutive slices of length size (the last may be shorter); ValueError if size <= 0.",
        [("range(0, len(items), size)", "range(0, len(items) - size, size)"), ("items[i : i + size]", "items[i : i + size - 1]")],
        lambda r: (list(range(r.randint(0, 12))), r.randint(1, 5)),
    ),
    "top_words": (
        '''def top_words(text, k):
    counts = {}
    for raw in text.split():
        word = raw.strip(".,;:!?\\"'()").lower()
        if word:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:k]]
''',
        'top_words(text, k): the k most frequent words, case-insensitive, ignoring surrounding punctuation .,;:!?"\'() ; ties in alphabetical order.',
        [("key=lambda item: (-item[1], item[0])", "key=lambda item: (item[1], item[0])"), (".lower()\n", "\n"), ("ranked[:k]", "ranked[: k - 1]")],
        lambda r: (" ".join(r.choice(["Alpha", "beta,", "gamma.", "alpha", "\"Beta\"", "delta!", "(gamma)", "beta", "Delta"]) for _ in range(r.randint(4, 14))), r.randint(1, 4)),
    ),
    "business_days": (
        '''import datetime as dt


def business_days(start, end, holidays=()):
    day = dt.date.fromisoformat(start)
    stop = dt.date.fromisoformat(end)
    off = {dt.date.fromisoformat(h) for h in holidays}
    count = 0
    while day < stop:
        if day.weekday() < 5 and day not in off:
            count += 1
        day += dt.timedelta(days=1)
    return count
''',
        "business_days(start, end, holidays=()): weekdays (Mon-Fri) in [start, end) that are not holidays; ISO date strings.",
        [("day.weekday() < 5", "day.weekday() < 6"), ("while day < stop", "while day <= stop"), ("day not in off", "True")],
        _business_case,
    ),
}


def _reference(name):
    scope = {}
    exec(FUNCTIONS[name][0], scope)
    return scope[name]


def _cases(name, rng, n):
    fn, make = _reference(name), FUNCTIONS[name][3]
    out, seen = [], set()
    for _ in range(n * 6):
        args = make(rng)
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        try:
            out.append((args, fn(*args), None))
        except ValueError:
            out.append((args, None, "ValueError"))
        if len(out) == n:
            break
    return out


def _detects(name, bug, cases):
    """True if the mutated function fails at least one case."""
    scope = {}
    try:
        exec(FUNCTIONS[name][0].replace(*bug), scope)
    except SyntaxError:
        return False
    fn = scope[name]
    for args, expected, err in cases:
        try:
            got = fn(*args)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ != err:
                return True
            continue
        if err or not _same(got, expected):
            return True
    return False


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 1e-9
    return a == b


def _test_file(name, cases, prefix):
    lines = []
    for i, (args, expected, err) in enumerate(cases, 1):
        call = f"lib.{name}({', '.join(map(repr, args))})"
        lines.append(f"def test_{prefix}{name}_{i}():")
        if err:
            lines += [
                "    try:",
                f"        got = {call}",
                "    except ValueError:",
                "        return",
                f"    raise AssertionError(f\"{call.replace(chr(34), chr(39))} returned {{got!r}}, expected ValueError\")",
            ]
        else:
            lines += [
                f"    got = {call}",
                f"    assert _same(got, {expected!r}), f\"{call.replace(chr(34), chr(39))} returned {{got!r}}, expected {expected!r}\"",
            ]
        lines.append("")
    return "\n".join(lines)


TEST_HEADER = '''import lib


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) < 1e-9
    return a == b


'''

RUNNER = r'''
import json, sys
sys.path.insert(0, ".")
sys.dont_write_bytecode = True
results = {}
try:
    import importlib
    mod = importlib.import_module(MODULE)
except Exception as e:
    print(json.dumps({"import_error": f"{type(e).__name__}: {e}"}))
    raise SystemExit
for name in sorted(n for n in dir(mod) if n.startswith("test_")):
    try:
        getattr(mod, name)()
        results[name] = "pass"
    except AssertionError as e:
        results[name] = "FAIL: " + str(e)[:200]
    except Exception as e:
        results[name] = f"ERROR: {type(e).__name__}: {str(e)[:150]}"
print(json.dumps({"results": results}))
'''


def _run_suite(ws, module):
    out = ws.run_python(f"MODULE = {module!r}\n{RUNNER}", limit=1_000_000)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"import_error": out[-300:]}


def make(difficulty, rng):
    # functions in lib.py, seeded bugs, and bugs that no visible test catches
    n_funcs, n_bugs, n_silent = {1: (2, 1, 0), 2: (3, 1, 0), 3: (5, 2, 1), 4: (6, 3, 2)}[difficulty]
    names = rng.sample(sorted(FUNCTIONS), n_funcs)
    buggy = rng.sample(names, n_bugs)
    bugs = {}
    hidden = {name: _cases(name, rng, 8) for name in names}
    for name in buggy:
        options = [b for b in FUNCTIONS[name][2] if _detects(name, b, hidden[name])]
        bugs[name] = rng.choice(options)

    imports = sorted({line for n in names for line in FUNCTIONS[n][0].splitlines() if line.startswith("import ")})

    def module(sources):
        """Join function sources into one file, hoisting their imports to the top."""
        sources = [re.sub(r"^import .*\n\n\n", "", b) for b in sources]
        return ("\n".join(imports) + "\n\n\n" if imports else "") + "\n\n".join(sources)

    lib = module([FUNCTIONS[n][0].replace(*bugs[n]) if n in bugs else FUNCTIONS[n][0] for n in names])
    fixed_lib = module([FUNCTIONS[n][0] for n in names])

    silent = set(rng.sample(buggy, n_silent))
    visible = []
    for name in names:
        if name in silent:  # two cases the bug gets right
            cases = [c for c in hidden[name] if not _detects(name, bugs[name], [c])][:2]
        else:
            cases = hidden[name][:3]
            if name in bugs and not _detects(name, bugs[name], cases):  # make sure one visible test fails
                cases = cases[:2] + [next(c for c in hidden[name] if _detects(name, bugs[name], [c]))]
        visible.append(_test_file(name, cases, ""))
    files = {
        "lib.py": lib,
        "test_lib.py": TEST_HEADER + "\n".join(visible),
        "SPEC.md": "# lib.py specification\n" + "\n".join(f"- {FUNCTIONS[n][1]}" for n in names) + "\n",
    }
    ws = Workspace(files)
    hidden_suite = TEST_HEADER + "\n".join(_test_file(n, hidden[n], "hidden_") for n in names)

    def check_path(args, verb):
        if arg(args, "path").strip() != "lib.py":
            raise ToolError(f"only lib.py can be {verb}")

    def save_lib(text):
        files["lib.py"] = text
        ws.write("lib.py", text)

    def do_edit(args):
        check_path(args, "edited")
        old, new = arg(args, "old"), arg(args, "new")
        count = files["lib.py"].count(old)
        if count != 1:
            raise ToolError(f'"old" must match exactly once in lib.py; it matches {count} times')
        save_lib(files["lib.py"].replace(old, new))
        if not new:
            return "edited lib.py"
        line = files["lib.py"][: files["lib.py"].index(new)].count("\n") + 1
        return f"edited lib.py at line {line}"

    def do_write(args):
        check_path(args, "written")
        save_lib(arg(args, "content"))
        return f"wrote lib.py ({len(files['lib.py'].splitlines())} lines)"

    def do_tests(args):
        report = _run_suite(ws, "test_lib")
        if "import_error" in report:
            return f"lib.py failed to import: {report['import_error']}"
        res = report["results"]
        failed = {k: v for k, v in res.items() if v != "pass"}
        lines = [f"{len(res) - len(failed)}/{len(res)} visible tests passed"]
        lines += [f"{k}: {v}" for k, v in list(failed.items())[:8]]
        return "\n".join(lines)

    def grade(answer):
        g = Workspace({"lib.py": files["lib.py"], "test_hidden.py": hidden_suite})
        try:
            report = _run_suite(g, "test_hidden")
        finally:
            g.close()
        if "import_error" in report:
            return 0.0, "lib.py does not import"
        res = report["results"]
        ok = {n: all(v == "pass" for k, v in res.items() if k.startswith(f"test_hidden_{n}_")) for n in names}
        if not all(ok[n] for n in names if n not in bugs):
            return 0.0, "a function that was correct is now broken"
        fixed = sum(ok[n] for n in bugs)
        return fixed / len(bugs), f"{fixed}/{len(bugs)} buggy functions fixed"

    tools = file_tools(ws, files, python=False) + [
        Tool("edit", '{"tool": "edit", "args": {"path": "lib.py", "old": "<exact text>", "new": "<replacement>"}}  replace text that occurs exactly once', do_edit),
        Tool("write", '{"tool": "write", "args": {"path": "lib.py", "content": "<whole file>"}}  replace lib.py', do_write),
        Tool("run_tests", '{"tool": "run_tests", "args": {}}  run test_lib.py against lib.py', do_tests),
    ]
    return Episode(
        question=f"lib.py has {n_bugs} bug{'s' if n_bugs > 1 else ''}: some functions do not match SPEC.md. Fix them without breaking the others. Hidden tests decide the grade.",
        tools=tools,
        answer_hint='When done: {"tool": "submit", "args": {"answer": {}}}',
        grade=grade,
        workspace=ws,
        oracle=[{"tool": "write", "args": {"path": "lib.py", "content": fixed_lib}}, submit_action({})],
    )


FAMILY = Family("code_repair", "software_engineering", "Fix seeded bugs in a small Python module against a spec and tests", make)
