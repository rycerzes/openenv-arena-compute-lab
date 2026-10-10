# SPDX-License-Identifier: BSD-3-Clause

"""Finance task families: loan schedules, FX ledgers, XNPV and XIRR.

Level 1 states everything plainly. Level 2 adds real-world formatting (accounting
amounts, day-first dates, units, case variants) that rules.md describes. Level 3
adds a memo.txt whose corrections override the data files.
"""

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal

from .base import AnswerField, Family, TaskInstance, accounting, cents, csv_text, day_first

CENT = Decimal("0.01")
MEMO_DECOYS = [
    "- Statements move to paperless delivery next quarter.",
    "- The audit committee meets on the second Tuesday.",
    "- Month-end close for the prior period is complete.",
]


def _dcents(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def _level_payment(balance: Decimal, monthly_rate: Decimal, months: int) -> Decimal:
    if monthly_rate == 0:
        return _dcents(balance / months)
    return _dcents(balance * monthly_rate / (1 - (1 + monthly_rate) ** -months))


def _memo(rng, lines):
    lines = list(lines) + [rng.choice(MEMO_DECOYS)]
    rng.shuffle(lines)
    return "# Corrections memo\n" + "\n".join(lines) + "\n"


def simulate_loan(principal, annual_rate, term, extras, rate_changes, deferrals):
    """Return per-month (balance_after, interest) following rules.md exactly."""
    bal = Decimal(principal)
    rate = Decimal(str(annual_rate)) / 1200
    pay = _level_payment(bal, rate, term)
    history = []
    month = 0
    while bal > 0 and month < 2000:
        month += 1
        if month in rate_changes:
            rate = Decimal(str(rate_changes[month])) / 1200
            pay = _level_payment(bal, rate, max(term - month + 1, 1))
        interest = _dcents(bal * rate)
        if month in deferrals:
            bal += interest
        else:
            bal = bal + interest - pay if bal + interest > pay else Decimal(0)
            bal = max(Decimal(0), bal - Decimal(extras.get(month, 0)))
        history.append((bal, interest))
    return history


LOAN_RULES = """# Loan rules
- Monthly rate r = annual_rate_percent / 100 / 12.
- Level payment for balance B over n months: B * r / (1 - (1 + r)^-n), rounded to the cent (half up).
- Each month m = 1, 2, ...: interest = balance * r rounded to the cent (half up).
  The payment covers that interest and the rest reduces the balance. If balance + interest
  is not more than the payment, the final payment clears the loan and the balance becomes 0.
- After the month's payment, an extra payment for month m (if any) reduces the balance (never below 0).
- Only extra payments whose status is applied count. Status values are case-insensitive.
- Amounts may carry a $ sign and thousands separators.
- A rate change at month m applies from month m: before charging that month's interest, the
  payment is recomputed as the level payment on the current balance over (term_months - m + 1) months.
- In a deferred month no payment and no extra payment are made; that month's interest is added to the balance.
- "Interest charged" for a month is the interest computed for that month (deferred months included).
- memo.txt, when present, corrects the other files and takes precedence over them.
"""


def _loan(level, rng):
    principal = rng.randrange(60, 400) * 1000
    annual = rng.randrange(300, 900, 5) / 100
    term = rng.choice([120, 180, 240, 300, 360])
    extras, extra_rows, rate_changes, deferrals = {}, [], {}, set()
    if level >= 2:
        months = sorted(rng.sample(range(2, 40), 6))
        statuses = ["applied"] * 4 + ["cancelled"] * 2
        rng.shuffle(statuses)
        for m, status in zip(months, statuses):
            amount = rng.randrange(5, 80) * 100
            if status == "applied":
                extras[m] = amount
                shown = rng.choice(["applied", "Applied", "APPLIED"])
            else:
                shown = rng.choice(["cancelled", "Cancelled", "reversed", "PENDING"])
            extra_rows.append([m, accounting(amount, symbol="$"), shown])
        change = rng.randrange(12, 30)
        rate_changes[change] = round(annual + rng.choice([-1, 1]) * rng.randrange(25, 200, 25) / 100, 2)

    files = {
        "loan.txt": f"principal_usd: {principal}\nannual_rate_percent: {annual}\nterm_months: {term}\n",
        "rules.md": LOAN_RULES,
    }
    if level >= 2:
        files["extra_payments.csv"] = csv_text(["month", "amount_usd", "status"], extra_rows)
        files["rate_changes.csv"] = csv_text(
            ["month", "annual_rate_percent"], [[m, r] for m, r in rate_changes.items()]
        )
    if level == 3:
        deferrals = set(rng.sample([m for m in range(3, 40) if m not in rate_changes and m - 1 not in rate_changes], 2))
        files["deferrals.csv"] = csv_text(["month"], [[m] for m in sorted(deferrals)])
        notes = []
        for kind in rng.sample(["bounce", "delay", "missing"], 2):
            if kind == "bounce":
                month = rng.choice(sorted(extras))
                del extras[month]
                notes.append(f"- The extra payment recorded for month {month} was returned unpaid. Treat it as cancelled.")
            elif kind == "delay":
                (month, rate), = rate_changes.items()
                rate_changes = {month + 1: rate}
                notes.append(f"- The rate change listed for month {month} took effect in month {month + 1} instead.")
            else:
                listed = {r[0] for r in extra_rows}
                month = rng.choice([m for m in range(2, 40) if m not in extras and m not in deferrals and m not in listed])
                amount = rng.randrange(20, 150) * 100
                extras[month] = amount
                notes.append(f"- An extra payment of ${amount:,.2f} was received in month {month} and is missing from extra_payments.csv.")
        files["memo.txt"] = _memo(rng, notes)
        history = simulate_loan(principal, annual, term, extras, rate_changes, deferrals)
        return TaskInstance(
            question="Simulate this loan to payoff following rules.md: applied extra payments, "
            "the rate change, deferred months and the memo's corrections.",
            files=files,
            fields=[
                AnswerField("payoff_month", len(history), "int", "the month in which the balance reaches 0"),
                AnswerField("total_interest", cents(float(sum(i for _, i in history))), "number", "interest charged over the whole loan, USD", abs_tol=2.0),
            ],
        )
    history = simulate_loan(principal, annual, term, extras, rate_changes, deferrals)
    k = rng.randrange(6, 30) if level == 1 else rng.randrange(max(rate_changes) + 2, 48)
    extra = " with the applied extra payments and the rate change" if level == 2 else ""
    return TaskInstance(
        question=f"Follow rules.md for this loan{extra}. Report the state after month {k}.",
        files=files,
        fields=[
            AnswerField("balance", cents(float(history[k - 1][0])), "number", f"balance after month {k}'s payments, USD", abs_tol=1.0),
            AnswerField("interest", cents(float(sum(i for _, i in history[:k]))), "number", f"interest charged in months 1-{k}, USD", abs_tol=1.0),
        ],
    )


FX_BASE = {"EUR": 1.08, "GBP": 1.27, "JPY": 0.0068, "CHF": 1.12, "CAD": 0.73, "AUD": 0.66}
FX_RULES = """# Ledger rules
- Only rows with status posted count.
- A posted row whose ref names another txn_id is a reversal: exclude both that row and the txn it names.
- Convert each counted row to USD at the rate of its date. If fx_rates.csv has no rate for that
  currency on that date, use the most recent earlier date that has one. USD rows use rate 1.
- A balance "as of" a date includes rows dated on or before that date.
- Sum the USD amounts and round the result to the cent at the end.
- Amounts may use thousands separators; an amount in parentheses is negative.
- Dates are either YYYY-MM-DD or day-first DD/MM/YYYY. Currency codes are case-insensitive.
- memo.txt, when present, corrects the other files and takes precedence over them.
"""


def _fx(level, rng):
    start = dt.date(2026, 3, 2)
    days = [start + dt.timedelta(d) for d in range(45)]
    ccys = rng.sample(sorted(FX_BASE), 3 if level == 1 else 4)
    rates, rate_rows = {}, []
    for c in ccys:
        value = FX_BASE[c] * rng.uniform(0.95, 1.05)
        for d in days:
            if d.weekday() >= 5:
                continue
            value *= 1 + rng.uniform(-0.006, 0.006)
            if level >= 2 and d != start and rng.random() < 0.12:
                continue
            q = float(f"{value:.5g}")
            rates[(d, c)] = q
            rate_rows.append([d.isoformat(), c, q])
    rate_rows.sort()

    accounts = ["OPS-01", "OPS-02", "TREASURY", "PAYROLL"]
    n = {1: 24, 2: 60, 3: 90}[level]
    rows = []
    for i in range(n):
        d = rng.choice(days if level >= 2 else [x for x in days if x.weekday() < 5])
        c = rng.choice(ccys + ["USD"])
        amount = rng.randrange(-90000, 120000) * 100 if c == "JPY" else rng.randrange(-900000, 1200000) / 100
        status = "posted" if level == 1 or rng.random() < 0.8 else rng.choice(["pending", "void"])
        rows.append([f"T{1000 + i}", d, rng.choice(accounts), c, amount, status, ""])
    if level >= 2:
        for orig in rng.sample([r for r in rows if r[5] == "posted"], 6):
            later = min(orig[1] + dt.timedelta(rng.randrange(1, 9)), days[-1])
            rows.append([f"T{1000 + len(rows)}", later, orig[2], orig[3], -orig[4], "posted", orig[0]])
    rows.sort(key=lambda r: (r[1], r[0]))
    reversed_ids = {r[6] for r in rows if r[5] == "posted" and r[6]}
    counted = [list(r) for r in rows if r[5] == "posted" and not r[6] and r[0] not in reversed_ids]
    as_of = rng.choice(days[25:])
    account = rng.choice(accounts)

    notes = []
    if level == 3:
        for kind in rng.sample(["feed", "misbooked"], rng.choice([1, 2])):
            if kind == "feed":
                def rate_dates_used(acct):
                    return sorted({r[1] for r in counted if r[3] != "USD" and (r[1], r[3]) in rates and r[1] <= as_of and r[1] != start and acct in (None, r[2])})

                bad = rng.choice(rate_dates_used(account) or rate_dates_used(None))
                for c in ccys:
                    rates.pop((bad, c), None)
                notes.append(f"- All fx_rates.csv rows dated {bad.isoformat()} came from a bad feed. Ignore them.")
            else:
                cands = [r for r in counted if r[1] <= as_of]
                row = rng.choice(cands)
                new = account if row[2] != account else rng.choice([a for a in accounts if a != account])
                notes.append(f"- {row[0]} was booked to {row[2]} by mistake; it belongs to {new}.")
                row[2] = new
        files_memo = _memo(rng, notes)

    def rate(d, c):
        if c == "USD":
            return 1.0
        while (d, c) not in rates:
            d -= dt.timedelta(1)
        return rates[(d, c)]

    usd = lambda r: r[4] * rate(r[1], r[3])
    balance = cents(sum(usd(r) for r in counted if r[2] == account and r[1] <= as_of))

    def show_date(d):
        return day_first(d) if level >= 2 and rng.random() < 0.4 else d.isoformat()

    def show_amount(a, c):
        return accounting(a, 0 if c == "JPY" else 2) if level >= 2 else a

    def show_ccy(c):
        return c.lower() if level == 3 and rng.random() < 0.3 else c

    files = {
        "transactions.csv": csv_text(
            ["txn_id", "date", "account", "currency", "amount", "status", "ref"],
            [[r[0], show_date(r[1]), r[2], show_ccy(r[3]), show_amount(r[4], r[3]), r[5], r[6]] for r in rows],
        ),
        "fx_rates.csv": csv_text(["date", "currency", "usd_per_unit"], rate_rows),
        "rules.md": FX_RULES,
    }
    fields = [AnswerField("balance_usd", balance, "number", f"USD balance of account {account} as of {as_of.isoformat()}", abs_tol=0.5)]
    if level == 3:
        files["memo.txt"] = files_memo
        exposure = {c: abs(sum(usd(r) for r in counted if r[3] == c and r[1] <= as_of)) for c in ccys}
        ranked = sorted(exposure, key=exposure.get, reverse=True)
        if exposure[ranked[0]] < 1.05 * exposure[ranked[1]]:
            return _fx(level, rng)
        fields.append(AnswerField("top_currency", ranked[0], "text", f"non-USD currency whose counted rows (all accounts, as of {as_of.isoformat()}) have the largest absolute USD sum; one of {', '.join(ccys)}", choices=ccys))
    return TaskInstance(
        question=f"Apply rules.md to the ledger. Account {account}, as of {as_of.isoformat()}.",
        files=files,
        fields=fields,
    )


def xnpv(rate, flows):
    d0 = flows[0][0]
    return sum(cf / (1 + rate) ** ((d - d0).days / 365) for d, cf in flows)


def xirr(flows):
    lo, hi = -0.95, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if xnpv(mid, flows) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


NPV_RULES = """# Valuation rules
- d0 is the date of the project's first cash flow (the earliest date).
- XNPV(r) = sum over cash flows of amount / (1 + r) ^ (days_since_d0 / 365).
- XIRR is the rate r at which XNPV(r) = 0.
- Report rates as percentages (12.34 means 12.34%).
- Amounts are in the unit given by the unit column: USD, or kUSD (thousands of USD).
- Dates are either YYYY-MM-DD or day-first DD/MM/YYYY.
- memo.txt, when present, corrects the other files and takes precedence over them.
"""


def _project(rng, regular, messy):
    start = dt.date(2026, 1, 1) + dt.timedelta(rng.randrange(0, 200))
    outlay = -rng.randrange(200, 2000) * 1000
    flows = [(start, outlay)]
    days = 0
    for i in range(1, rng.randrange(5, 9) + 1):
        days += rng.randrange(120, 420)
        d = dt.date(start.year + i, start.month, start.day) if regular else start + dt.timedelta(days)
        amount = round(-outlay * rng.uniform(0.12, 0.32), 2)
        if messy and rng.random() < 0.4:
            amount = float(round(amount))
        flows.append((d, amount))
    return flows


def _flow_row(rng, d, amount, messy):
    shown_date = day_first(d) if messy and rng.random() < 0.4 else d.isoformat()
    whole = abs(amount - round(amount)) < 1e-6
    if messy and whole and rng.random() < 0.8:
        return [shown_date, f"{amount / 1000:.3f}", "kUSD"]
    return [shown_date, f"{amount:.2f}", "USD"]


def _npv(level, rng):
    rate_pct = rng.randrange(600, 1300, 25) / 100
    r = rate_pct / 100
    files = {"rules.md": NPV_RULES}
    if level < 3:
        flows = _project(rng, regular=level == 1, messy=level == 2)
        files["cashflows.csv"] = csv_text(["date", "amount", "unit"], [_flow_row(rng, d, a, level == 2) for d, a in flows])
        fields = [AnswerField("npv", round(xnpv(r, flows), 2), "number", f"XNPV at {rate_pct}%, USD", abs_tol=1.0)]
        if level == 2:
            fields.append(AnswerField("irr_percent", round(100 * xirr(flows), 4), "number", "XIRR as a percentage", abs_tol=0.01))
        return TaskInstance(f"Value the project in cashflows.csv following rules.md, discount rate {rate_pct}%.", files, fields)

    names = rng.sample(["ALDER", "BIRCH", "CEDAR", "DOGWOOD", "ELM", "FIR"], 3)
    projects = {p: _project(rng, regular=False, messy=True) for p in names}
    rows = []
    for p, flows in projects.items():
        for d, a in flows:
            rows.append([p, "base", *_flow_row(rng, d, a, True)])
            down = a if a < 0 else round(a * rng.uniform(0.6, 0.9), 2)
            rows.append([p, "downside", *_flow_row(rng, d, down, True)])
    notes = []
    for kind in rng.sample(["slip", "outlay"], rng.choice([1, 2])):
        p = rng.choice(names)
        flows = projects[p]
        if kind == "slip":
            i = rng.randrange(1, len(flows))
            d, a = flows[i]
            new = d + dt.timedelta(rng.randrange(60, 200))
            flows[i] = (new, a)
            notes.append(f"- {p}: the base-scenario cash flow dated {d.isoformat()} slipped to {new.isoformat()}.")
        else:
            d, a = flows[0]
            new = round(a * rng.uniform(1.1, 1.35), -3)
            flows[0] = (d, new)
            notes.append(f"- {p}: the base-scenario initial outlay was revised to {new:.2f} USD.")
        flows.sort()
    npvs = {p: xnpv(r, f) for p, f in projects.items()}
    ranked = sorted(npvs, key=npvs.get, reverse=True)
    if npvs[ranked[0]] - npvs[ranked[1]] < 0.02 * abs(npvs[ranked[0]]) + 1000:
        return _npv(level, rng)
    rng.shuffle(rows)
    files["cashflows.csv"] = csv_text(["project", "scenario", "date", "amount", "unit"], rows)
    files["memo.txt"] = _memo(rng, notes)
    best = ranked[0]
    return TaskInstance(
        f"Use only the base scenario in cashflows.csv. Discount rate {rate_pct}%. Follow rules.md.",
        files,
        [
            AnswerField("best_project", best, "text", f"project with the highest base-scenario XNPV; one of {', '.join(names)}", choices=names),
            AnswerField("best_npv", round(npvs[best], 2), "number", "that project's XNPV, USD", abs_tol=1.0),
            AnswerField("best_irr_percent", round(100 * xirr(projects[best]), 4), "number", "that project's XIRR as a percentage", abs_tol=0.01),
        ],
    )


FAMILIES = [
    Family("loan_schedule", "finance", "Amortization with extra payments, rate resets, deferrals and corrections", _loan),
    Family("fx_ledger", "finance", "Multi-currency ledger with reversals, missing FX dates, messy formats and corrections", _fx),
    Family("xnpv_xirr", "finance", "Irregular-date XNPV and XIRR with mixed units, project ranking and corrections", _npv),
]
