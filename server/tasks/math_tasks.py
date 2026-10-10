# SPDX-License-Identifier: BSD-3-Clause

"""Math and formal-reasoning task families: counting, shortest paths, recurrences."""

import heapq
from functools import lru_cache

from .base import AnswerField, Family, TaskInstance, csv_text

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23]


def digit_sum(n: int) -> int:
    return sum(map(int, str(n)))


def count_brute(lo, hi, m, r, q=None, s=None, banned=()):
    total = 0
    for n in range(lo + (r - lo) % m, hi + 1, m):
        if any(n % p == 0 for p in banned):
            continue
        if q is not None and digit_sum(n) % q != s:
            continue
        total += 1
    return total


def count_digit_dp(hi, m, r, q, s, banned_digit=None):
    """Count 0 <= n <= hi with n % m == r, digit_sum(n) % q == s and no digit equal to banned_digit (1-9)."""
    if hi < 0:
        return 0
    digits = list(map(int, str(hi)))

    @lru_cache(maxsize=None)
    def go(pos, tight, mod_m, mod_q):
        if pos == len(digits):
            return int(mod_m == r and mod_q == s)
        top = digits[pos] if tight else 9
        return sum(
            go(pos + 1, tight and d == top, (mod_m * 10 + d) % m, (mod_q + d) % q)
            for d in range(top + 1)
            if d != banned_digit
        )

    return go(0, True, 0, 0)


def _remainder(what, r, m):
    return f"{what} is a multiple of {m}" if r == 0 else f"{what} leaves remainder {r} when divided by {m}"


def _count(level, rng):
    m = rng.randrange(3, 13)
    r = rng.randrange(m)
    if level == 1:
        lo = rng.randrange(1, 5000)
        hi = lo + rng.randrange(20000, 80000)
        banned = sorted(rng.sample([p for p in PRIMES if m % p], 2))
        expected = count_brute(lo, hi, m, r, banned=banned)
        lines = [f"n % {m} == {r}", f"n is not divisible by any of {', '.join(map(str, banned))}"]
    elif level == 2:
        lo = rng.randrange(1, 10**5)
        hi = lo + rng.randrange(10**6, 3 * 10**6)
        q = rng.randrange(4, 10)
        s = rng.randrange(q)
        banned = sorted(rng.sample([p for p in PRIMES if m % p], 2))
        expected = count_brute(lo, hi, m, r, q, s, banned)
        lines = [
            _remainder("n", r, m),
            f"n is a multiple of neither {banned[0]} nor {banned[1]}",
            _remainder("the sum of the decimal digits of n", s, q),
        ]
        rng.shuffle(lines)
    else:
        lo = rng.randrange(10**11, 5 * 10**11)
        hi = lo + rng.randrange(10**12, 9 * 10**12)
        q = rng.randrange(4, 10)
        s = rng.randrange(q)
        digit = rng.randrange(1, 10)
        expected = count_digit_dp(hi, m, r, q, s, digit) - count_digit_dp(lo - 1, m, r, q, s, digit)
        lines = [
            _remainder("n", r, m),
            _remainder("the sum of the decimal digits of n", s, q),
            f"the digit {digit} does not appear anywhere in n",
        ]
        rng.shuffle(lines)
    spec = f"range: {lo} <= n <= {hi}\nconditions (all must hold):\n" + "".join(f"- {line}\n" for line in lines)
    return TaskInstance(
        "Count the integers described in spec.txt.",
        {"spec.txt": spec},
        [AnswerField("count", expected, "int", "how many integers n satisfy spec.txt")],
    )


def dijkstra(adj, src):
    """Distances and shortest-path counts from src."""
    dist, ways = {src: 0}, {src: 1}
    heap, done = [(0, src)], set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v], ways[v] = nd, ways[u]
                heapq.heappush(heap, (nd, v))
            elif nd == dist[v]:
                ways[v] += ways[u]
    return dist, ways


def _duration(minutes, rng):
    if minutes >= 60 and rng.random() < 0.6:
        h, m = divmod(minutes, 60)
        return f"{h} h {m:02d} min" if m else f"{h} h"
    return f"{minutes} min"


def _graph(level, rng):
    n, m = {1: (20, 70), 2: (60, 200), 3: (150, 450)}[level]
    weights = list({1: range(1, 21), 2: range(5, 125), 3: range(15, 105, 15)}[level])
    nodes = [f"N{i}" for i in range(1, n + 1)]
    edges = {}
    while len(edges) < m:
        u, v = rng.sample(nodes, 2)
        if (v, u) not in edges:
            edges[(u, v)] = rng.choice(weights)
    rows = []
    for (u, v), w in edges.items():
        two_way = level >= 2 and rng.random() < 0.35
        status = "open" if level == 1 or rng.random() > 0.1 else "closed"
        rows.append([u, v, w, "two-way" if two_way else "one-way", status])
    adj = {}
    for u, v, w, direction, status in rows:
        if status == "open":
            adj.setdefault(u, []).append((v, w))
            if direction == "two-way":
                adj.setdefault(v, []).append((u, w))
    src = rng.choice(nodes)
    dist, ways = dijkstra(adj, src)
    far = [x for x in dist if x != src and dist[x] >= 3 * min(weights)]
    if len(far) < 2:
        return _graph(level, rng)
    if level == 3:
        dst = max(far, key=lambda x: (ways[x], x))
        if ways[dst] < 2:
            return _graph(level, rng)
    else:
        dst = rng.choice(far)
    unit = "" if level == 1 else " in minutes"
    fields = [AnswerField("cost", dist[dst], "int", f"cost{unit} of the cheapest route from {src} to {dst}")]
    if level == 2:
        via = rng.choice([x for x in dist if x not in (src, dst) and dst in dijkstra(adj, x)[0]])
        fields.append(AnswerField("cost_via", dist[via] + dijkstra(adj, via)[0][dst], "int", f"cost in minutes of the cheapest route from {src} to {dst} that visits {via}"))
    if level == 3:
        fields.append(AnswerField("num_cheapest_routes", ways[dst], "int", f"number of distinct cheapest routes from {src} to {dst}"))
    if level == 1:
        text = csv_text(["from", "to", "cost"], [r[:3] for r in rows])
        question = f"edges.csv lists one-way roads. Routes go from {src} to {dst}."
    else:
        text = csv_text(["from", "to", "travel_time", "direction", "status"], [[u, v, _duration(w, rng), d, st] for u, v, w, d, st in rows])
        question = f"edges.csv lists roads with travel times. Two-way roads can be used in both directions; closed roads cannot be used. Routes go from {src} to {dst}."
    return TaskInstance(question, {"edges.csv": text}, fields)


def recurrence_loop(a0, a1, c1, c2, c3, d, mod, n_max):
    a_prev, a = a0 % mod, a1 % mod
    total = (a_prev + a) % mod
    if n_max == 0:
        return a_prev, a_prev
    for n in range(2, n_max + 1):
        a_prev, a = a, (c1 * a + c2 * a_prev + c3 * n + d) % mod
        total = (total + a) % mod
    return a, total


def _matmul(x, y, mod):
    size = len(x)
    return [[sum(x[i][k] * y[k][j] for k in range(size)) % mod for j in range(size)] for i in range(size)]


def _apply_power(step, power, state, mod):
    """step^power @ state, mod `mod`, by repeated squaring."""
    size = len(step)
    result = [[int(i == j) for j in range(size)] for i in range(size)]
    while power:
        if power & 1:
            result = _matmul(result, step, mod)
        step = _matmul(step, step, mod)
        power >>= 1
    return [sum(result[i][j] * state[j] for j in range(size)) % mod for i in range(size)]


def recurrence_fast(a0, a1, c1, c2, c3, d, mod, n_max):
    """Matrix power over the state [a_n, a_(n-1), n, 1, S_n]."""
    if n_max < 2:
        return recurrence_loop(a0, a1, c1, c2, c3, d, mod, n_max)
    step = [
        [c1, c2, c3, c3 + d, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [c1, c2, c3, c3 + d, 1],
    ]
    state = [a1 % mod, a0 % mod, 1, 1, (a0 + a1) % mod]
    out = _apply_power(step, n_max - 1, state, mod)
    return out[0], out[4]


def coupled_loop(a0, b0, p, q, r, s, u, v, mod, n_max):
    a, b = a0 % mod, b0 % mod
    for n in range(1, n_max + 1):
        a, b = (p * a + q * b + u) % mod, (r * a + s * b + v * n) % mod
    return a, b


def coupled_fast(a0, b0, p, q, r, s, u, v, mod, n_max):
    """Matrix power over the state [a_n, b_n, n, 1]."""
    step = [[p, q, 0, u], [r, s, v, v], [0, 0, 1, 1], [0, 0, 0, 1]]
    out = _apply_power(step, n_max, [a0 % mod, b0 % mod, 0, 1], mod)
    return out[0], out[1]


def _recurrence(level, rng):
    mod = rng.choice([10**9 + 7, 998244353, rng.randrange(10**6, 10**9)])
    if level == 3:
        a0, b0 = rng.randrange(mod), rng.randrange(mod)
        p, q, r, s, u, v = (rng.randrange(1, 1000) for _ in range(6))
        n_max = rng.randrange(10**14, 10**16)
        a_n, b_n = coupled_fast(a0, b0, p, q, r, s, u, v, mod, n_max)
        spec = (
            f"a_0 = {a0}\nb_0 = {b0}\nfor n >= 1, using the previous terms:\n"
            f"  a_n = ({p} * a_(n-1) + {q} * b_(n-1) + {u}) mod {mod}\n"
            f"  b_n = ({r} * a_(n-1) + {s} * b_(n-1) + {v} * n) mod {mod}\nN = {n_max}\n"
        )
        fields = [AnswerField("a_N", a_n, "int", f"a_N for N = {n_max}"), AnswerField("b_N", b_n, "int", f"b_N for N = {n_max}")]
        return TaskInstance("Evaluate the sequences defined in sequence.txt.", {"sequence.txt": spec}, fields)

    c1, c2, c3, d = (rng.randrange(1, 1000) for _ in range(4))
    x, y = rng.randrange(mod), rng.randrange(mod)
    if level == 1:
        n_max = rng.randrange(2000, 9000)
        a_n, _ = recurrence_fast(x, y, c1, c2, c3, d, mod, n_max)
        spec = f"a_0 = {x}\na_1 = {y}\nfor n >= 2: a_n = ({c1} * a_(n-1) + {c2} * a_(n-2) + {c3} * n + {d}) mod {mod}\nN = {n_max}\n"
        return TaskInstance("Evaluate the sequence defined in sequence.txt.", {"sequence.txt": spec}, [AnswerField("a_N", a_n, "int", f"a_N for N = {n_max}")])
    # 1-indexed: a_1 = x, a_2 = y. With b_k = a_(k+1) this is the 0-indexed recurrence with d' = c3 + d.
    n_max = rng.randrange(10**6, 4 * 10**6)
    a_n, s_n = recurrence_fast(x, y, c1, c2, c3, c3 + d, mod, n_max - 1)
    spec = (
        f"a_1 = {x}\na_2 = {y}\nfor n >= 3: a_n = ({c1} * a_(n-1) + {c2} * a_(n-2) + {c3} * n + {d}) mod {mod}\n"
        f"S_N = (a_1 + a_2 + ... + a_N) mod {mod}\nN = {n_max}\n"
    )
    fields = [AnswerField("a_N", a_n, "int", f"a_N for N = {n_max}"), AnswerField("S_N", s_n, "int", f"S_N for N = {n_max}")]
    return TaskInstance("Evaluate the sequence defined in sequence.txt.", {"sequence.txt": spec}, fields)


FAMILIES = [
    Family("constrained_count", "math", "Counting integers under modular, digit-sum and forbidden-digit constraints", _count),
    Family("graph_paths", "math", "Shortest routes with mixed time formats, two-way roads, waypoints and route counting", _graph),
    Family("recurrence_mod", "math", "Linear and coupled recurrences mod M, 1-indexed variants, astronomically large N", _recurrence),
]
