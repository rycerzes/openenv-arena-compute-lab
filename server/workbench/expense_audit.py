# SPDX-License-Identifier: BSD-3-Clause

"""Office work: audit one employee's expense claims against a written policy, using SQL.

The claims, city tiers and FX rates are tables the agent queries with a read-only
`sql` tool (no python). Higher difficulty adds foreign currencies and duplicate
claims, then weekend rules and non-reimbursable categories, then a memo that
amends the policy from a given date. The answer is the reimbursable total and
the set of fully rejected claims; both must be right.
"""

import datetime as dt
import json

from ..sandbox import Workspace
from ..tasks.base import cents, csv_text
from .base import Episode, Family, Tool, ToolError, arg, file_tools, submit_action, submit_hint

EMPLOYEES = ["A. Rivera", "B. Okafor", "C. Lindqvist", "D. Tanaka", "E. Haddad", "F. Moreau"]
CITIES = {"London": "A", "New York": "A", "Tokyo": "A", "Berlin": "B", "Toronto": "B", "Madrid": "B", "Porto": "C", "Austin": "C", "Leeds": "C"}
MEAL_CAP = {"A": 75.0, "B": 60.0, "C": 45.0}
LODGING_CAP = {"A": 250.0, "B": 180.0, "C": 120.0}
RATES = {"USD": 1.0, "EUR": 1.0841, "GBP": 1.2718, "JPY": 0.006812, "CAD": 0.7334}
RECEIPT_LIMIT = 75.0

POLICY = """# Expense policy
1. Convert every claim to USD at rates.csv (usd_per_unit) and round to the cent (half up).
2. A claim is rejected outright (reimbursed 0) if any of these hold:
   - its category is supplies (supplies go through procurement);
   - it has no receipt and its USD amount is above the receipt limit ($75.00);
   - it is dated on a Saturday or Sunday, unless its note contains "approved travel";
   - it duplicates an earlier claim (same employee, date, category, amount and currency);
     the claim with the lowest claim_id counts, the later ones are rejected.
3. Caps on claims that are not rejected, by the tier of the claim's city (tiers.csv):
   - meals: the employee's meal claims for one day together are reimbursed up to the meal cap
     (tier A $75, tier B $60, tier C $45);
   - lodging: each lodging claim is reimbursed up to the nightly cap (A $250, B $180, C $120);
   - taxi and airfare are not capped.
4. The reimbursable total is the sum after rejections and caps, rounded to the cent.
5. memo.txt, when present, amends this policy for claims dated on or after its effective date.
"""

SQL = r'''
import csv, json, sqlite3
args = json.loads(ARGS)
con = sqlite3.connect(":memory:")
for name in args["tables"]:
    with open(name + ".csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0])
    def num(v):
        try:
            return float(v)
        except ValueError:
            return None
    kinds = {c: "REAL" if all(num(r[c]) is not None for r in rows) else "TEXT" for c in cols}
    con.execute(f"CREATE TABLE {name} ({', '.join(f'{c} {kinds[c]}' for c in cols)})")
    con.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(cols))})",
                    [[num(r[c]) if kinds[c] == "REAL" else r[c] for c in cols] for r in rows])
con.execute("PRAGMA query_only = ON")
try:
    cur = con.execute(args["query"])
except sqlite3.Error as e:
    print(f"SQL error: {e}"); raise SystemExit
names = [d[0] for d in cur.description or []]
rows = cur.fetchall()
print(" | ".join(names))
for r in rows[:40]:
    print(" | ".join("" if v is None else (f"{v:g}" if isinstance(v, float) else str(v)) for v in r))
print(f"[{len(rows)} rows" + ("; showing first 40]" if len(rows) > 40 else "]"))
'''


def make(difficulty, rng):
    n_claims = {1: 8, 2: 14, 3: 24, 4: 34}[difficulty]
    employee = rng.choice(EMPLOYEES)
    others = [e for e in EMPLOYEES if e != employee]
    currencies = ["USD"] if difficulty == 1 else ["USD", *rng.sample(["EUR", "GBP", "JPY", "CAD"], 2)]
    start = dt.date(2026, 9, 1) + dt.timedelta(rng.randrange(0, 40))
    days = [start + dt.timedelta(d) for d in range(14)]
    if difficulty < 3:
        days = [d for d in days if d.weekday() < 5]
    city_of_day = {d: rng.choice(sorted(CITIES)) for d in days}

    claims = []
    for i in range(n_claims):
        who = employee if i % 5 in (0, 2) or rng.random() < 0.3 else rng.choice(others)
        day = rng.choice(days)
        category = rng.choice(["meals", "meals", "meals", "lodging", "taxi", "airfare"] + (["supplies"] if difficulty >= 3 else []))
        currency = rng.choice(currencies)
        usd = {"meals": rng.uniform(12, 70), "lodging": rng.uniform(90, 320), "taxi": rng.uniform(8, 60), "airfare": rng.uniform(150, 900), "supplies": rng.uniform(10, 120)}[category]
        amount = round(usd / RATES[currency], 0 if currency == "JPY" else 2)
        receipt = "yes" if rng.random() < 0.8 else "no"
        note = ""
        if day.weekday() >= 5 and rng.random() < 0.5:
            note = "approved travel"
        elif rng.random() < 0.15:
            note = rng.choice(["client dinner", "conference", "late flight"])
        claims.append({"claim_id": f"C{1001 + i}", "employee": who, "date": day, "category": category,
                       "city": city_of_day[day], "amount": amount, "currency": currency, "receipt": receipt, "note": note})
    if difficulty >= 2:
        mine = [c for c in claims if c["employee"] == employee]
        for c in rng.sample(mine, min(2, len(mine))):
            claims.append({**c, "claim_id": f"C{1001 + len(claims)}", "note": ""})

    amend = None
    if difficulty == 4:
        when = rng.choice(days[3:10])
        tier = rng.choice(["A", "B", "C"])
        if rng.random() < 0.5:
            amend = ("meal_cap", when, tier, MEAL_CAP[tier] + rng.choice([-15.0, 10.0, 15.0]))
        else:
            amend = ("receipt_limit", when, None, rng.choice([40.0, 50.0]))

    def usd_of(c):
        return cents(c["amount"] * RATES[c["currency"]])

    def receipt_limit(c):
        return amend[3] if amend and amend[0] == "receipt_limit" and c["date"] >= amend[1] else RECEIPT_LIMIT

    def meal_cap(c):
        tier = CITIES[c["city"]]
        if amend and amend[0] == "meal_cap" and c["date"] >= amend[1] and tier == amend[2]:
            return amend[3]
        return MEAL_CAP[tier]

    seen, rejected, kept = set(), [], []
    for c in sorted(claims, key=lambda c: int(c["claim_id"][1:])):
        key = (c["employee"], c["date"], c["category"], c["amount"], c["currency"])
        dup = key in seen
        seen.add(key)
        if c["employee"] != employee:
            continue
        reject = (
            c["category"] == "supplies"
            or (c["receipt"] == "no" and usd_of(c) > receipt_limit(c))
            or (c["date"].weekday() >= 5 and "approved travel" not in c["note"])
            or dup
        )
        (rejected if reject else kept).append(c)
    total = 0.0
    meals_by_day = {}
    for c in kept:
        if c["category"] == "meals":
            meals_by_day.setdefault(c["date"], []).append(c)
        elif c["category"] == "lodging":
            total += min(usd_of(c), LODGING_CAP[CITIES[c["city"]]])
        else:
            total += usd_of(c)
    for day, items in meals_by_day.items():
        total += min(sum(usd_of(c) for c in items), meal_cap(items[0]))
    total = cents(total)
    rejected_ids = sorted(c["claim_id"] for c in rejected)

    rows = [[c["claim_id"], c["employee"], c["date"].isoformat(), c["category"], c["city"], c["amount"], c["currency"], c["receipt"], c["note"]] for c in claims]
    rng.shuffle(rows)
    files = {
        "policy.md": POLICY,
        "claims.csv": csv_text(["claim_id", "employee", "date", "category", "city", "amount", "currency", "receipt", "note"], rows),
        "tiers.csv": csv_text(["city", "tier"], sorted(CITIES.items())),
        "rates.csv": csv_text(["currency", "usd_per_unit"], [[c, RATES[c]] for c in sorted(set(currencies))]),
    }
    if amend:
        if amend[0] == "meal_cap":
            line = f"- Effective {amend[1].isoformat()}, the tier {amend[2]} meal cap is ${amend[3]:.2f} per day."
        else:
            line = f"- Effective {amend[1].isoformat()}, receipts are required for claims above ${amend[3]:.2f}."
        files["memo.txt"] = "# Finance memo\n" + line + "\n- The office will be closed on the first Monday of next month.\n"
    ws = Workspace(files)
    tables = ["claims", "tiers", "rates"]

    def sql(args):
        query = arg(args, "query")
        if len(query) > 1500:
            raise ToolError("query too long")
        return ws.run_python(f"ARGS = {json.dumps({'query': query, 'tables': tables})!r}\n{SQL}", limit=4000)

    def grade(answer):
        if not isinstance(answer, dict):
            return 0.0, 'answer needs "total_usd" and "rejected"'
        try:
            got_total = float(str(answer.get("total_usd")).replace("$", "").replace(",", ""))
        except ValueError:
            return 0.0, "total_usd is not a number"
        got_rejected = answer.get("rejected")
        if not isinstance(got_rejected, list):
            return 0.0, "rejected must be a list of claim ids"
        total_ok = abs(got_total - total) <= 0.01
        rejected_ok = sorted(str(x).strip().upper() for x in got_rejected) == rejected_ids
        return float(total_ok and rejected_ok), f"total {'right' if total_ok else 'wrong'}, rejected list {'right' if rejected_ok else 'wrong'}"

    hint = (
        "Answer fields:\n- total_usd: reimbursable total for " + employee + " in USD\n"
        "- rejected: claim ids of " + employee + "'s claims that are rejected outright\n"
        + submit_hint({"total_usd": 0.0, "rejected": ["C1001"]})
    )
    return Episode(
        question=f"Audit the expense claims of {employee} following policy.md. Tables claims, tiers and rates can be queried with sql.",
        tools=[t for t in file_tools(ws, files, python=False)] + [
            Tool("sql", '{"tool": "sql", "args": {"query": "SELECT ... FROM claims"}}  read-only SQLite over tables claims, tiers, rates', sql)
        ],
        answer_hint=hint,
        grade=grade,
        workspace=ws,
        oracle=[submit_action({"total_usd": total, "rejected": rejected_ids})],
    )


FAMILY = Family("expense_audit", "office_white_collar", "Audit expense claims against a written policy with SQL: caps, receipts, weekends, duplicates, FX and amendments", make)
