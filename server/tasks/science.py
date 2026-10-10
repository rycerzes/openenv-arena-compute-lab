# SPDX-License-Identifier: BSD-3-Clause

"""Natural-science task families: kinetics fits, projectile ODEs, stoichiometry."""

import math
import re

from .base import AnswerField, Family, TaskInstance, csv_text

R_GAS = 8.314
G = 9.81


def _sig(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def linfit(xs, ys):
    """Least-squares line: (slope, intercept, r_squared)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    return slope, my - slope * mx, (sxy * sxy) / (sxx * syy)


TRANSFORMS = {0: lambda c: c, 1: math.log, 2: lambda c: 1 / c}


def fit_order(ts, cs):
    """Pick the order whose linearized plot has the highest R^2; return (order, slope, intercept)."""
    fits = {o: linfit(ts, [f(c) for c in cs]) for o, f in TRANSFORMS.items()}
    order = max(fits, key=lambda o: fits[o][2])
    return order, fits[order][0], fits[order][1]


def _conc(order, k, c0, t):
    if order == 0:
        return c0 - k * t
    if order == 1:
        return c0 * math.exp(-k * t)
    return 1 / (1 / c0 + k * t)


def _k_for(order, c0, frac, t_end):
    if order == 0:
        return c0 * (1 - frac) / t_end
    if order == 1:
        return -math.log(frac) / t_end
    return (1 / (c0 * frac) - 1 / c0) / t_end


def _run(rng, order, k, c0, t_end, points, messy):
    """Measurements as shown in the file plus the SI values they stand for."""
    if not messy:
        ts = [round(t_end * i / (points - 1)) for i in range(points)]
        cs = [_sig(_conc(order, k, c0, t)) for t in ts]
        return csv_text(["t_s", "conc_M"], list(zip(ts, cs))), ts, cs
    t_min = sorted({0.0, round(t_end / 60, 1), *(round(rng.uniform(0, t_end / 60), 1) for _ in range(points - 2))})
    rows, ts, cs = [], [], []
    for t in t_min:
        c_mM = _sig(_conc(order, k, c0, t * 60) * 1000 * (1 + rng.gauss(0, 0.004)))
        rows.append([t, c_mM, "ok"])
        ts.append(t * 60)
        cs.append(c_mM / 1000)
    for _ in range(2):
        t = round(rng.uniform(0, t_end / 60), 1)
        rows.append([t, _sig(_conc(order, k, c0, t * 60) * 1000 * rng.choice([0.7, 1.35])), "fail"])
    rows.sort()
    return csv_text(["t_min", "conc_mM", "qc"], rows), ts, cs


KINETICS_RULES = """# Fitting rules
- Work in seconds and mol/L (M). 1 min = 60 s; 1 mM = 0.001 M.
- Rows whose qc column is fail are instrument faults: leave them out.
- For each candidate order fit a least-squares line against time t (s):
  order 0: [A] vs t; order 1: ln[A] vs t; order 2: 1/[A] vs t.
- The reaction order is the candidate whose line has the highest R^2.
- The rate constant k is the absolute value of that line's slope
  (units M/s, 1/s or 1/(M*s) for orders 0, 1, 2).
- Predictions use the fitted line (slope and intercept), not the first data point.
- Arrhenius: k = A * exp(-Ea / (R * T)), R = 8.314 J/(mol*K), T in kelvin (K = C + 273.15).
"""


def _kinetics(level, rng):
    order = rng.choice([0, 1, 2])
    c0 = round(rng.uniform(0.4, 2.0), 3)
    t_end = rng.randrange(300, 4000, 10)
    frac = rng.uniform(0.15, 0.35)
    k = _k_for(order, c0, frac, t_end)
    messy = level >= 2
    text, ts, cs = _run(rng, order, k, c0, t_end, 10 if level == 1 else 12, messy)
    fitted, slope, intercept = fit_order(ts, cs)
    if fitted != order:
        return _kinetics(level, rng)
    files = {"rules.md": KINETICS_RULES}
    if level < 3:
        files["measurements.csv"] = text
        fields = [
            AnswerField("order", order, "int", "reaction order: 0, 1 or 2"),
            AnswerField("k", abs(slope), "number", "rate constant k in M and s units (see rules.md)", rel_tol=0.01),
        ]
        if level == 2:
            target_mM = _sig(cs[-1] * 1000 * rng.uniform(0.6, 0.9), 3)
            t_target = (TRANSFORMS[order](target_mM / 1000) - intercept) / slope
            fields.append(AnswerField("t_target_s", t_target, "number", f"time in s at which the fitted line gives [A] = {target_mM} mM", rel_tol=0.01))
        return TaskInstance("Determine the kinetics of the reaction in measurements.csv following rules.md.", files, fields)

    t1 = rng.randrange(15, 40)
    t2 = t1 + rng.randrange(10, 30)
    ea = rng.uniform(40, 110) * 1000
    k2 = k * math.exp(-ea / R_GAS * (1 / (t2 + 273.15) - 1 / (t1 + 273.15)))
    t_end2 = max(60, round(t_end * k / k2, -1))
    text2, ts2, cs2 = _run(rng, order, k2, c0, t_end2, 12, messy)
    if min(cs2) <= 0:  # a zero-order run B can pass zero after t_end2 is rounded up
        return _kinetics(level, rng)
    fitted2, slope2, _ = fit_order(ts2, cs2)
    if fitted2 != order:
        return _kinetics(level, rng)
    ka, kb = abs(slope), abs(slope2)
    ea_fit = R_GAS * math.log(kb / ka) / (1 / (t1 + 273.15) - 1 / (t2 + 273.15)) / 1000
    files["run_A.csv"] = text
    files["run_B.csv"] = text2
    files["conditions.txt"] = f"run_A temperature: {t1} C\nrun_B temperature: {t2 + 273.15:.2f} K\n"
    return TaskInstance(
        "The same reaction was run at two temperatures (run_A.csv, run_B.csv, conditions.txt). Follow rules.md.",
        files,
        [
            AnswerField("order", order, "int", "reaction order: 0, 1 or 2"),
            AnswerField("ea_kj_per_mol", ea_fit, "number", "activation energy from the two fitted k values, kJ/mol", rel_tol=0.01),
        ],
    )


PROJECTILE_RULES = """# Simulation rules
- Work in SI units: 1 km/h = 1/3.6 m/s, 1 g = 0.001 kg, 1 ft = 0.3048 m, 1 knot = 0.514444 m/s.
- State: position (x, y) in m, velocity (vx, vy) in m/s. Launch at x = 0 at the launch height,
  with the launch speed at angle_deg above horizontal. g = 9.81 m/s^2.
- Wind w (along +x) depends on the current altitude y (see wind.csv when present; otherwise w = 0).
  Bands are [from, to); above the last band use the last band's wind; below the first use the first.
- Air velocity: ux = vx - w, uy = vy; u = sqrt(ux^2 + uy^2).
- Acceleration: ax = -(drag_k / mass_kg) * u * ux; ay = -g - (drag_k / mass_kg) * u * uy.
- Explicit Euler with step dt_s, all from the current state:
  x' = x + vx*dt; y' = y + vy*dt; vx' = vx + ax*dt; vy' = vy + ay*dt; t' = t + dt.
- Stop at the first step whose new y' is below 0. Landing point and time are the linear
  interpolation between the last two states at y = 0.
- Max height is the largest y among the computed states before landing (launch state included).
"""


def simulate_projectile(speed, angle, mass, drag_k, dt, h0=0.0, bands=None):
    def wind(y):
        if not bands:
            return 0.0
        for lo, hi, w in bands:
            if lo <= y < hi:
                return w
        return bands[0][2] if y < bands[0][0] else bands[-1][2]

    x, y = 0.0, float(h0)
    vx, vy = speed * math.cos(math.radians(angle)), speed * math.sin(math.radians(angle))
    steps, ymax = 0, y
    while True:
        ux, uy = vx - wind(y), vy
        u = math.hypot(ux, uy)
        ax = -(drag_k / mass) * u * ux
        ay = -G - (drag_k / mass) * u * uy
        nx, ny = x + vx * dt, y + vy * dt
        vx, vy = vx + ax * dt, vy + ay * dt
        if ny < 0:
            frac = y / (y - ny)
            return x + frac * (nx - x), (steps + frac) * dt, ymax
        x, y = nx, ny
        steps += 1
        ymax = max(ymax, y)


def _projectile(level, rng):
    angle = rng.randrange(25, 66)
    v_term = rng.uniform(25, 80)
    dt = 0.01 if level != 2 else 0.005
    bands = None
    files = {"rules.md": PROJECTILE_RULES}
    if level == 1:
        speed, mass, h0 = rng.randrange(30, 90), round(rng.uniform(0.1, 2.0), 3), 0
        drag_k = _sig(mass * G / v_term**2)
        params = f"speed_m_s: {speed}\nangle_deg: {angle}\nmass_kg: {mass}\ndrag_k_kg_per_m: {drag_k}\ndt_s: {dt}\nlaunch_height_m: {h0}\n"
    else:
        speed_kmh, mass_g, h0_ft = round(rng.uniform(100, 320), 1), rng.randrange(100, 2000), rng.randrange(0, 80)
        speed, mass, h0 = speed_kmh / 3.6, mass_g / 1000, h0_ft * 0.3048
        drag_k = _sig(mass * G / v_term**2)
        params = f"speed_km_h: {speed_kmh}\nangle_deg: {angle}\nmass_g: {mass_g}\ndrag_k_kg_per_m: {drag_k}\ndt_s: {dt}\nlaunch_height_ft: {h0_ft}\n"
    if level == 3:
        edges = [0, *sorted(rng.sample(range(20, 400, 20), 3)), 1300]
        shown = [(edges[i], edges[i + 1], rng.randrange(-30, 31)) for i in range(4)]
        bands = [(lo * 0.3048, hi * 0.3048, w * 0.514444) for lo, hi, w in shown]
        files["wind.csv"] = csv_text(["from_ft", "to_ft", "wind_knots"], [list(b) for b in shown])
    files["launch.txt"] = params
    rng_x, t_land, ymax = simulate_projectile(speed, angle, mass, drag_k, dt, h0, bands)
    fields = [AnswerField("range_m", rng_x, "number", "landing x, m", rel_tol=0.001)]
    if level >= 2:
        fields.append(AnswerField("flight_time_s", t_land, "number", "landing time, s", rel_tol=0.001))
    if level == 2:
        fields.append(AnswerField("max_height_m", ymax, "number", "max height, m", rel_tol=0.001))
    return TaskInstance("Simulate the projectile in launch.txt following rules.md.", files, fields)


ATOMIC_WEIGHTS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Cu": 63.546, "Zn": 65.38, "Ag": 107.87, "Ba": 137.33,
}
ATOMIC_WEIGHTS_CSV = csv_text(["element", "atomic_weight"], sorted(ATOMIC_WEIGHTS.items()))

# (reactants, products) as (coefficient, formula) pairs; all balanced (checked in tests).
REACTIONS = [
    ([(2, "H2"), (1, "O2")], [(2, "H2O")]),
    ([(1, "N2"), (3, "H2")], [(2, "NH3")]),
    ([(1, "CH4"), (2, "O2")], [(1, "CO2"), (2, "H2O")]),
    ([(1, "Fe2O3"), (3, "CO")], [(2, "Fe"), (3, "CO2")]),
    ([(2, "Al"), (3, "Cl2")], [(2, "AlCl3")]),
    ([(1, "C3H8"), (5, "O2")], [(3, "CO2"), (4, "H2O")]),
    ([(4, "Fe"), (3, "O2")], [(2, "Fe2O3")]),
    ([(1, "Zn"), (2, "HCl")], [(1, "ZnCl2"), (1, "H2")]),
    ([(2, "Na"), (1, "Cl2")], [(2, "NaCl")]),
    ([(1, "Ca(OH)2"), (2, "HCl")], [(1, "CaCl2"), (2, "H2O")]),
    ([(2, "Al"), (3, "H2SO4")], [(1, "Al2(SO4)3"), (3, "H2")]),
    ([(1, "C2H5OH"), (3, "O2")], [(2, "CO2"), (3, "H2O")]),
    ([(2, "KOH"), (1, "H2SO4")], [(1, "K2SO4"), (2, "H2O")]),
    ([(1, "Mg"), (2, "HCl")], [(1, "MgCl2"), (1, "H2")]),
    ([(4, "NH3"), (5, "O2")], [(4, "NO"), (6, "H2O")]),
    ([(2, "C8H18"), (25, "O2")], [(16, "CO2"), (18, "H2O")]),
    ([(1, "AgNO3"), (1, "NaCl")], [(1, "AgCl"), (1, "NaNO3")]),
    ([(3, "NO2"), (1, "H2O")], [(2, "HNO3"), (1, "NO")]),
    ([(1, "BaCl2"), (1, "Na2SO4")], [(1, "BaSO4"), (2, "NaCl")]),
]


def parse_formula(formula: str) -> dict[str, int]:
    tokens = re.findall(r"[A-Z][a-z]?|\d+|\(|\)", formula)
    stack = [{}]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok == "(":
            stack.append({})
            continue
        if tok == ")":
            group = stack.pop()
            mult = 1
            if i < len(tokens) and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            for el, n in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + n * mult
            continue
        n = 1
        if i < len(tokens) and tokens[i].isdigit():
            n = int(tokens[i])
            i += 1
        stack[-1][tok] = stack[-1].get(tok, 0) + n
    return stack[0]


def molar_mass(formula: str) -> float:
    return sum(ATOMIC_WEIGHTS[el] * n for el, n in parse_formula(formula).items())


def _equation(reaction) -> str:
    side = lambda terms: " + ".join(f"{c} {f}" if c > 1 else f for c, f in terms)
    return f"{side(reaction[0])} -> {side(reaction[1])}"


STOICH_RULES = """# Lab rules
- Molar masses come from atomic_weights.csv (sum of atom counts times weights).
- Masses are in the unit given by the unit column (g, mg or kg); work in grams.
- Purity is a percentage when written with %, otherwise a fraction (0.925 = 92.5%).
  Effective mass = mass * purity (purity 100% when no purity is given).
- Moles of a reagent = effective mass / molar mass.
- The limiting reagent is the one with the smallest moles / stoichiometric coefficient.
- Theoretical yield of a product = (limiting moles / its coefficient) * product coefficient * product molar mass.
- Percent yield = actual yield / theoretical yield * 100.
- In a two-step synthesis, the amount of intermediate carried into step 2 is the step-1 theoretical
  yield times step1_yield_percent / 100. Step-2 reagents are pure.
- memo.txt, when present, corrects the other files and takes precedence over them.
"""


def _charge(rng, reaction, margin=0.08):
    """Gram masses for the two reagents, with a clear limiting one."""
    (ca, fa), (cb, fb) = reaction[0]
    extent = rng.uniform(0.05, 0.6)
    excess = rng.uniform(1 + margin, 1.8)
    limit_first = rng.random() < 0.5
    na = ca * extent * (1 if limit_first else excess)
    nb = cb * extent * (excess if limit_first else 1)
    return {fa: na * molar_mass(fa), fb: nb * molar_mass(fb)}


def _show_mass(rng, grams, messy):
    """(shown value, unit, grams it stands for)."""
    if not messy:
        g = round(grams, 2)
        return g, "g", g
    unit = rng.choice(["g", "mg", "kg"])
    if unit == "mg":
        v = round(grams * 1000)
        return v, unit, v / 1000
    if unit == "kg":
        v = round(grams / 1000, 6)
        return v, unit, v * 1000
    g = round(grams, 2)
    return g, unit, g


def _show_purity(rng):
    pct = round(rng.uniform(85, 99.5), 1)
    return (f"{pct}%" if rng.random() < 0.5 else f"{pct / 100:.3f}"), pct


def _limiting(reaction, grams):
    coef = {f: c for c, f in reaction[0]}
    ext = {f: grams[f] / molar_mass(f) / coef[f] for f in coef}
    lim = min(ext, key=ext.get)
    other = max(ext.values())
    return lim, ext[lim], other / ext[lim]


def _stoich(level, rng):
    files = {"rules.md": STOICH_RULES, "atomic_weights.csv": ATOMIC_WEIGHTS_CSV}
    messy = level >= 2
    if level < 3:
        reaction = rng.choice(REACTIONS)
        rows, grams = [], {}
        for f, g in _charge(rng, reaction).items():
            v, unit, g_shown = _show_mass(rng, g, messy)
            if level == 2:
                shown_p, pct = _show_purity(rng)
                rows.append([f, v, unit, shown_p])
                grams[f] = g_shown * pct / 100
            else:
                rows.append([f, v, unit])
                grams[f] = g_shown
        lim, extent, margin = _limiting(reaction, grams)
        if margin < 1.03:
            return _stoich(level, rng)
        cp, product = rng.choice(reaction[1])
        theo = extent * cp * molar_mass(product)
        header = ["reagent", "mass", "unit"] + (["purity"] if level == 2 else [])
        files["batch.csv"] = csv_text(header, rows)
        files["reaction.txt"] = _equation(reaction) + "\n"
        choices = [f for _, f in reaction[0]]
        fields = [
            AnswerField("limiting", lim, "text", f"limiting reagent formula, one of {', '.join(choices)}", choices=choices),
            AnswerField("theoretical_g", theo, "number", f"theoretical yield of {product}, g", rel_tol=0.005),
        ]
        if level == 2:
            actual = round(theo * rng.uniform(0.55, 0.95), 2)
            files["result.txt"] = f"actual_yield_g of {product}: {actual}\n"
            fields.append(AnswerField("percent_yield", actual / theo * 100, "number", "percent yield", abs_tol=0.3))
        return TaskInstance("Work out the batch in batch.csv for the reaction in reaction.txt, following rules.md.", files, fields)

    chains = [(a, b, f) for a in REACTIONS for b in REACTIONS if a is not b for _, f in a[1] if f in {g for _, g in b[0]}]
    r1, r2, inter = rng.choice(chains)
    rows, grams1 = [], {}
    for f, g in _charge(rng, r1).items():
        v, unit, g_shown = _show_mass(rng, g, True)
        rows.append([1, f, v, unit])
        grams1[f] = g_shown
    _, ext1, margin1 = _limiting(r1, grams1)
    c_int = next(c for c, f in r1[1] if f == inter)
    y1 = round(rng.uniform(60, 95), 1)
    (c_a, f_a), (c_b, f_b) = r2[0]
    other, c_other = (f_b, c_b) if f_a == inter else (f_a, c_a)
    c_inter2 = c_a if f_a == inter else c_b
    ratio = rng.choice([rng.uniform(0.55, 0.9), rng.uniform(1.1, 1.6)])
    v, unit, g_other = _show_mass(rng, ext1 * c_int * y1 / 100 / c_inter2 * ratio * c_other * molar_mass(other), True)
    rows.append([2, other, v, unit])
    notes = []
    kind = rng.choice(["yield", "mass"])
    if kind == "yield":
        y_new = round(min(99.0, max(40.0, y1 + rng.choice([-1, 1]) * rng.uniform(8, 20))), 1)
        notes.append(f"- Step 1 yield was re-measured: use step1_yield_percent = {y_new}.")
        y_used, g_other_used = y_new, g_other
    else:
        g_new = round(g_other * rng.choice([rng.uniform(0.5, 0.8), rng.uniform(1.25, 1.6)]), 2)
        notes.append(f"- The step 2 {other} mass was mis-recorded; the correct value is {g_new} g.")
        y_used, g_other_used = y1, g_new
    ext_inter = ext1 * c_int * y_used / 100 / c_inter2
    ext_other = g_other_used / molar_mass(other) / c_other
    if max(ext_inter, ext_other) / min(ext_inter, ext_other) < 1.03 or margin1 < 1.03:
        return _stoich(level, rng)
    lim2, ext2 = (inter, ext_inter) if ext_inter < ext_other else (other, ext_other)
    cp, product = rng.choice([(c, f) for c, f in r2[1] if f != inter])
    files["reaction.txt"] = f"step 1: {_equation(r1)}\nstep 2: {_equation(r2)}\n"
    files["batch.csv"] = csv_text(["step", "reagent", "mass", "unit"], rows)
    files["process.txt"] = f"intermediate: {inter}\nstep1_yield_percent: {y1}\nstep 2 uses all intermediate carried over plus the step 2 reagent in batch.csv\n"
    files["memo.txt"] = "# Corrections memo\n" + "\n".join(notes) + "\n- Fume hood 3 is booked for calibration on Friday.\n"
    choices = [inter, other]
    return TaskInstance(
        "Two-step synthesis (reaction.txt, batch.csv, process.txt). Follow rules.md.",
        files,
        [
            AnswerField("limiting_step2", lim2, "text", f"limiting reagent of step 2, one of {', '.join(choices)}", choices=choices),
            AnswerField("product_g", ext2 * cp * molar_mass(product), "number", f"theoretical step-2 yield of {product}, g", rel_tol=0.005),
        ],
    )


FAMILIES = [
    Family("reaction_kinetics", "natural_science", "Order and rate-constant fits, Arrhenius activation energy", _kinetics),
    Family("projectile_ode", "natural_science", "Euler integration of a projectile with quadratic drag and wind", _projectile),
    Family("stoichiometry", "natural_science", "Limiting reagents, yields, purity and two-step synthesis", _stoich),
]
