# SPDX-License-Identifier: BSD-3-Clause

"""Reference solvers that see only what the agent sees: the prompt and the task files.

Each solver is Python source run through the environment's own `python` tool,
with the prompt bound to `PROMPT`. It prints the answer as JSON.
"""

COMMON = r"""
import csv, json, math, os, re
def rows(name):
    with open(name) as f:
        return list(csv.DictReader(f))
def kv(name):
    out = {}
    for line in open(name).read().splitlines():
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out
def field_names():
    return re.findall(r"^- (\w+):", PROMPT, re.M)
def num(text):
    text = str(text).strip().replace("$", "").replace(",", "")
    return -float(text[1:-1]) if text.startswith("(") else float(text)
def day(text):
    import datetime as dt
    return dt.datetime.strptime(text, "%d/%m/%Y").date() if "/" in text else dt.date.fromisoformat(text)
memo = open("memo.txt").read() if os.path.exists("memo.txt") else ""
"""

LOAN = r"""
from decimal import Decimal, ROUND_HALF_UP
D = Decimal
q = lambda x: x.quantize(D("0.01"), rounding=ROUND_HALF_UP)
info = kv("loan.txt")
bal, r, term = D(info["principal_usd"]), D(info["annual_rate_percent"]) / 1200, int(info["term_months"])
level_pay = lambda b, r, n: q(b * r / (1 - (1 + r) ** -n))
extras = {int(x["month"]): D(str(num(x["amount_usd"]))) for x in rows("extra_payments.csv") if x["status"].lower() == "applied"} if os.path.exists("extra_payments.csv") else {}
changes = {int(x["month"]): D(x["annual_rate_percent"]) / 1200 for x in rows("rate_changes.csv")} if os.path.exists("rate_changes.csv") else {}
defer = {int(x["month"]) for x in rows("deferrals.csv")} if os.path.exists("deferrals.csv") else set()
for mo in re.findall(r"recorded for month (\d+) was returned", memo):
    extras.pop(int(mo), None)
for old, new in re.findall(r"listed for month (\d+) took effect in month (\d+)", memo):
    changes[int(new)] = changes.pop(int(old))
for amt, mo in re.findall(r"extra payment of \$([\d,.]+) was received in month (\d+)", memo):
    extras[int(mo)] = D(amt.replace(",", ""))
pay = level_pay(bal, r, term)
hist = []
m = 0
while bal > 0:
    m += 1
    if m in changes:
        r = changes[m]
        pay = level_pay(bal, r, term - m + 1)
    i = q(bal * r)
    if m in defer:
        bal += i
    else:
        bal = bal + i - pay if bal + i > pay else D(0)
        bal = max(D(0), bal - extras.get(m, 0))
    hist.append((bal, i))
if "payoff_month" in field_names():
    ans = {"payoff_month": len(hist), "total_interest": float(sum(i for _, i in hist))}
else:
    k = int(re.search(r"after month (\d+)", PROMPT).group(1))
    ans = {"balance": float(hist[k - 1][0]), "interest": float(sum(i for _, i in hist[:k]))}
print(json.dumps(ans))
"""

FX = r"""
import datetime as dt
acct, as_of = re.search(r"Account (\S+), as of (\d{4}-\d\d-\d\d)", PROMPT).groups()
as_of = dt.date.fromisoformat(as_of)
bad = {dt.date.fromisoformat(d) for d in re.findall(r"rows dated (\S+) came from a bad feed", memo)}
moves = dict((t, new) for t, new in re.findall(r"(T\d+) was booked to \S+ by mistake; it belongs to (\S+)\.", memo))
rates = {(day(x["date"]), x["currency"].strip().upper()): float(x["usd_per_unit"]) for x in rows("fx_rates.csv") if day(x["date"]) not in bad}
def rate(d, c):
    if c == "USD":
        return 1.0
    while (d, c) not in rates:
        d -= dt.timedelta(1)
    return rates[(d, c)]
tx = rows("transactions.csv")
for t in tx:
    t["date"] = day(t["date"]); t["currency"] = t["currency"].strip().upper(); t["amount"] = num(t["amount"])
    t["account"] = moves.get(t["txn_id"], t["account"])
posted = [t for t in tx if t["status"].lower() == "posted"]
gone = {t["ref"] for t in posted if t["ref"]} | {t["txn_id"] for t in posted if t["ref"]}
live = [t for t in posted if t["txn_id"] not in gone and t["date"] <= as_of]
usd = lambda t: t["amount"] * rate(t["date"], t["currency"])
ans = {"balance_usd": round(sum(usd(t) for t in live if t["account"] == acct), 2)}
if "top_currency" in field_names():
    tot = {}
    for t in live:
        if t["currency"] != "USD":
            tot[t["currency"]] = tot.get(t["currency"], 0) + usd(t)
    ans["top_currency"] = max(tot, key=lambda c: abs(tot[c]))
print(json.dumps(ans))
"""

NPV = r"""
import datetime as dt
r = float(re.search(r"[Dd]iscount rate ([\d.]+)%", PROMPT).group(1)) / 100
def xnpv(rate, flows):
    d0 = min(d for d, _ in flows)
    return sum(a / (1 + rate) ** ((d - d0).days / 365) for d, a in flows)
def xirr(flows):
    lo, hi = -0.9, 5.0
    for _ in range(300):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if xnpv(mid, flows) > 0 else (lo, mid)
    return lo
projects = {}
for x in rows("cashflows.csv"):
    if x.get("scenario", "base") == "base":
        amount = float(x["amount"]) * (1000 if x["unit"] == "kUSD" else 1)
        projects.setdefault(x.get("project", "P"), []).append([day(x["date"]), amount])
for p, old, new in re.findall(r"(\w+): the base-scenario cash flow dated (\S+) slipped to (\S+)\.", memo):
    for f in projects[p]:
        if f[0] == dt.date.fromisoformat(old):
            f[0] = dt.date.fromisoformat(new)
for p, new in re.findall(r"(\w+): the base-scenario initial outlay was revised to (\S+) USD", memo):
    min(projects[p])[1] = float(new)
names = field_names()
if "best_project" in names:
    best = max(projects, key=lambda p: xnpv(r, projects[p]))
    ans = {"best_project": best, "best_npv": xnpv(r, projects[best]), "best_irr_percent": 100 * xirr(projects[best])}
else:
    flows = projects["P"]
    ans = {"npv": xnpv(r, flows)}
    if "irr_percent" in names:
        ans["irr_percent"] = 100 * xirr(flows)
print(json.dumps(ans))
"""

KINETICS = r"""
def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx, my - sxy / sxx * mx, sxy * sxy / (sxx * syy)
tf = {0: lambda c: c, 1: math.log, 2: lambda c: 1 / c}
def analyze(name):
    data = [x for x in rows(name) if x.get("qc", "ok") != "fail"]
    if "t_s" in data[0]:
        t = [float(x["t_s"]) for x in data]; c = [float(x["conc_M"]) for x in data]
    else:
        t = [float(x["t_min"]) * 60 for x in data]; c = [float(x["conc_mM"]) / 1000 for x in data]
    fits = {o: fit(t, [f(v) for v in c]) for o, f in tf.items()}
    o = max(fits, key=lambda k: fits[k][2])
    return o, fits[o]
def kelvin(text):
    v, unit = text.split()
    return float(v) + (273.15 if unit == "C" else 0)
names = field_names()
if "ea_kj_per_mol" in names:
    temps = kv("conditions.txt")
    oa, fa = analyze("run_A.csv"); ob, fb = analyze("run_B.csv")
    ta, tb = kelvin(temps["run_A temperature"]), kelvin(temps["run_B temperature"])
    ea = 8.314 * math.log(abs(fb[0]) / abs(fa[0])) / (1 / ta - 1 / tb) / 1000
    ans = {"order": oa, "ea_kj_per_mol": ea}
else:
    o, (slope, icpt, _) = analyze("measurements.csv")
    ans = {"order": o, "k": abs(slope)}
    if "t_target_s" in names:
        target = float(re.search(r"\[A\] = ([\d.eE+-]+) mM", PROMPT).group(1)) / 1000
        ans["t_target_s"] = (tf[o](target) - icpt) / slope
print(json.dumps(ans))
"""

PROJECTILE = r"""
p = {k: float(v) for k, v in kv("launch.txt").items()}
speed = p["speed_m_s"] if "speed_m_s" in p else p["speed_km_h"] / 3.6
mass = p["mass_kg"] if "mass_kg" in p else p["mass_g"] / 1000
h0 = p["launch_height_m"] if "launch_height_m" in p else p["launch_height_ft"] * 0.3048
bands = [(float(x["from_ft"]) * 0.3048, float(x["to_ft"]) * 0.3048, float(x["wind_knots"]) * 0.514444) for x in rows("wind.csv")] if os.path.exists("wind.csv") else []
def wind(y):
    if not bands:
        return 0.0
    if y < bands[0][0]:
        return bands[0][2]
    for lo, hi, w in bands:
        if lo <= y < hi:
            return w
    return bands[-1][2]
g, k, dt = 9.81, p["drag_k_kg_per_m"], p["dt_s"]
a = math.radians(p["angle_deg"])
x, y, vx, vy = 0.0, h0, speed * math.cos(a), speed * math.sin(a)
n, top = 0, y
while True:
    ux = vx - wind(y); u = math.sqrt(ux * ux + vy * vy)
    ax = -(k / mass) * u * ux; ay = -g - (k / mass) * u * vy
    x2, y2 = x + vx * dt, y + vy * dt
    vx, vy = vx + ax * dt, vy + ay * dt
    if y2 < 0:
        f = y / (y - y2)
        land_x, land_t = x + f * (x2 - x), (n + f) * dt
        break
    x, y, n = x2, y2, n + 1
    top = max(top, y)
ans = {"range_m": land_x, "flight_time_s": land_t, "max_height_m": top}
print(json.dumps({k: ans[k] for k in field_names()}))
"""

STOICH = r"""
W = {x["element"]: float(x["atomic_weight"]) for x in rows("atomic_weights.csv")}
def mm(formula):
    toks = re.findall(r"[A-Z][a-z]?|\d+|\(|\)", formula)
    stack = [0.0]; i = 0
    while i < len(toks):
        t = toks[i]; i += 1
        if t == "(":
            stack.append(0.0); continue
        mult = 1
        if i < len(toks) and toks[i].isdigit():
            mult = int(toks[i]); i += 1
        if t == ")":
            g = stack.pop(); stack[-1] += g * mult
        else:
            stack[-1] += W[t] * mult
    return stack[0]
def side(s):
    out = []
    for term in s.split(" + "):
        parts = term.strip().split()
        out.append((int(parts[0]), parts[1]) if len(parts) == 2 else (1, parts[0]))
    return out
def eq(line):
    l, r = line.split("->")
    return side(l), side(r)
def extent(reactants, moles):
    return min((moles[f] / c, f) for c, f in reactants if f in moles)
def grams(x):
    g = float(x["mass"]) * {"g": 1, "mg": 0.001, "kg": 1000}[x["unit"]]
    pur = (x.get("purity") or "100%").strip()
    return g * (float(pur[:-1]) / 100 if pur.endswith("%") else float(pur))
names = field_names()
if "limiting_step2" in names:
    lines = open("reaction.txt").read().splitlines()
    r1 = eq(lines[0].split(":", 1)[1]); r2 = eq(lines[1].split(":", 1)[1])
    proc = kv("process.txt")
    inter = proc["intermediate"]; y1 = float(proc["step1_yield_percent"])
    m = re.search(r"use step1_yield_percent = (\d+(?:\.\d+)?)", memo)
    if m:
        y1 = float(m.group(1))
    b = rows("batch.csv")
    mol1 = {x["reagent"]: grams(x) / mm(x["reagent"]) for x in b if x["step"] == "1"}
    e1, _ = extent(r1[0], mol1)
    c_int = next(c for c, f in r1[1] if f == inter)
    mol2 = {x["reagent"]: grams(x) / mm(x["reagent"]) for x in b if x["step"] == "2"}
    for f, g in re.findall(r"step 2 (\S+) mass was mis-recorded; the correct value is ([\d.]+) g", memo):
        mol2[f] = float(g) / mm(f)
    mol2[inter] = e1 * c_int * y1 / 100
    e2, lim = extent(r2[0], mol2)
    prod = re.search(r"yield of (\S+), g", PROMPT).group(1)
    cp = next(c for c, f in r2[1] if f == prod)
    ans = {"limiting_step2": lim, "product_g": e2 * cp * mm(prod)}
else:
    reactants, products = eq(open("reaction.txt").read().strip())
    mol = {x["reagent"]: grams(x) / mm(x["reagent"]) for x in rows("batch.csv")}
    e, lim = extent(reactants, mol)
    prod = re.search(r"yield of (\S+), g", PROMPT).group(1)
    cp = next(c for c, f in products if f == prod)
    theo = e * cp * mm(prod)
    ans = {"limiting": lim, "theoretical_g": theo}
    if "percent_yield" in names:
        actual = float(open("result.txt").read().split(":")[1])
        ans["percent_yield"] = actual / theo * 100
print(json.dumps(ans))
"""

COUNT = r"""
from functools import lru_cache
spec = open("spec.txt").read()
lo, hi = map(int, re.search(r"range: (\d+) <= n <= (\d+)", spec).groups())
def rem(what):
    m = re.search(what + r" (?:% (\d+) == (\d+)|leaves remainder (\d+) when divided by (\d+)|is a multiple of (\d+))", spec)
    if not m:
        return None
    g = m.groups()
    if g[0]:
        return int(g[0]), int(g[1])
    if g[2]:
        return int(g[3]), int(g[2])
    return int(g[4]), 0
m, r = rem(r"- n")
ds = rem(r"the sum of the decimal digits of n")
ban = re.search(r"divisible by any of ([\d, ]+)", spec) or re.search(r"multiple of neither (\d+) nor (\d+)", spec)
ban = [int(v) for v in re.findall(r"\d+", ban.group(0))] if ban else []
nodig = re.search(r"the digit (\d) does not appear", spec)
nodig = int(nodig.group(1)) if nodig else None
def upto(N, q, s):
    if N < 0:
        return 0
    dig = list(map(int, str(N)))
    @lru_cache(None)
    def go(i, tight, a, b):
        if i == len(dig):
            return int(a == r and b == s)
        return sum(go(i + 1, tight and d == dig[i], (a * 10 + d) % m, (b + d) % q) for d in range((dig[i] if tight else 9) + 1) if d != nodig)
    return go(0, True, 0, 0)
if ds and not ban:
    q, s = ds
    ans = upto(hi, q, s) - upto(lo - 1, q, s)
else:
    ans = 0
    for n in range(lo + (r - lo) % m, hi + 1, m):
        if all(n % p for p in ban) and (not ds or sum(map(int, str(n))) % ds[0] == ds[1]):
            ans += 1
print(json.dumps({"count": ans}))
"""

GRAPH = r"""
import heapq
src, dst = re.search(r"from (N\d+) to (N\d+)", PROMPT).groups()
def minutes(text):
    h = re.search(r"(\d+) h", text); mm_ = re.search(r"(\d+) min", text)
    return (int(h.group(1)) * 60 if h else 0) + (int(mm_.group(1)) if mm_ else 0)
adj = {}
for e in rows("edges.csv"):
    if e.get("status", "open") != "open":
        continue
    w = int(e["cost"]) if "cost" in e else minutes(e["travel_time"])
    adj.setdefault(e["from"], []).append((e["to"], w))
    if e.get("direction") == "two-way":
        adj.setdefault(e["to"], []).append((e["from"], w))
def sp(s):
    dist, cnt, pq, seen = {s: 0}, {s: 1}, [(0, s)], set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        for v, w in adj.get(u, []):
            if v not in dist or d + w < dist[v]:
                dist[v] = d + w; cnt[v] = cnt[u]; heapq.heappush(pq, (d + w, v))
            elif d + w == dist[v]:
                cnt[v] += cnt[u]
    return dist, cnt
dist, cnt = sp(src)
ans = {"cost": dist[dst]}
names = field_names()
if "cost_via" in names:
    via = re.search(r"that visits (N\d+)", PROMPT).group(1)
    ans["cost_via"] = dist[via] + sp(via)[0][dst]
if "num_cheapest_routes" in names:
    ans["num_cheapest_routes"] = cnt[dst]
print(json.dumps(ans))
"""

RECURRENCE = r"""
s = open("sequence.txt").read()
N = int(re.search(r"^N = (\d+)", s, re.M).group(1))
def mul(A, B, M):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) % M for j in range(n)] for i in range(n)]
def power_apply(T, e, v, M):
    n = len(T)
    R = [[int(i == j) for j in range(n)] for i in range(n)]
    while e:
        if e & 1:
            R = mul(R, T, M)
        T = mul(T, T, M); e >>= 1
    return [sum(R[i][j] * v[j] for j in range(n)) % M for i in range(n)]
if "b_0" in s:
    a0 = int(re.search(r"a_0 = (\d+)", s).group(1)); b0 = int(re.search(r"b_0 = (\d+)", s).group(1))
    p, q, u, M = map(int, re.search(r"a_n = \((\d+) \* a_\(n-1\) \+ (\d+) \* b_\(n-1\) \+ (\d+)\) mod (\d+)", s).groups())
    r_, s_, v = map(int, re.search(r"b_n = \((\d+) \* a_\(n-1\) \+ (\d+) \* b_\(n-1\) \+ (\d+) \* n\)", s).groups())
    # state (a, b, n, 1): b_(n+1) uses v * (n + 1)
    T = [[p, q, 0, u], [r_, s_, v, v], [0, 0, 1, 1], [0, 0, 0, 1]]
    out = power_apply(T, N, [a0, b0, 0, 1], M)
    ans = {"a_N": out[0], "b_N": out[1]}
else:
    c1, c2, c3, d, M = map(int, re.search(r"\((\d+) \* a_\(n-1\) \+ (\d+) \* a_\(n-2\) \+ (\d+) \* n \+ (\d+)\) mod (\d+)", s).groups())
    one_indexed = "a_2 =" in s
    if one_indexed:
        x0, x1, start = int(re.search(r"a_1 = (\d+)", s).group(1)), int(re.search(r"a_2 = (\d+)", s).group(1)), 2
    else:
        x0, x1, start = int(re.search(r"a_0 = (\d+)", s).group(1)), int(re.search(r"a_1 = (\d+)", s).group(1)), 1
    # state (a_k, a_(k-1), k, 1, S_k) for k = start; a_(k+1) uses c3 * (k + 1)
    T = [[c1, c2, c3, c3 + d, 0], [1, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 0], [c1, c2, c3, c3 + d, 1]]
    out = power_apply(T, N - start, [x1 % M, x0 % M, start, 1, (x0 + x1) % M], M)
    ans = {"a_N": out[0], "S_N": out[4]}
print(json.dumps({k: ans[k] for k in field_names()}))
"""

SOLVERS = {
    "loan_schedule": LOAN,
    "fx_ledger": FX,
    "xnpv_xirr": NPV,
    "reaction_kinetics": KINETICS,
    "projectile_ode": PROJECTILE,
    "stoichiometry": STOICH,
    "constrained_count": COUNT,
    "graph_paths": GRAPH,
    "recurrence_mod": RECURRENCE,
}


def solver_code(family: str, prompt: str) -> str:
    return f"PROMPT = {prompt!r}\n{COMMON}\n{SOLVERS[family]}"
