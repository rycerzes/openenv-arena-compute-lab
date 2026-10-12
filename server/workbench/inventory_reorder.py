# SPDX-License-Identifier: BSD-3-Clause

"""Industrial and physical systems: decide today's purchase orders from a reorder policy.

Stock, forecasts, supplier terms and open purchase orders are CSV files. Higher
difficulty adds open orders (some cancelled or arriving too late to count),
then pack sizes and minimum order quantities, then a memo that changes lead
times or discontinues items. The answer maps each SKU to order to its quantity.
"""

import datetime as dt
import math

from ..sandbox import Workspace
from ..tasks.base import csv_text
from .base import Episode, Family, file_tools, submit_action, submit_hint

POLICY = """# Reorder policy (review today)
- Inventory position = on_hand - reserved + quantity of open purchase orders with status
  confirmed and eta on or before today + lead_time_days. Other open orders do not count.
- Safety stock = ceil(daily_forecast * SAFETY_DAYS) units.
- Reorder point = daily_forecast * lead_time_days + safety stock.
- Order only SKUs whose inventory position is at or below their reorder point.
- Order quantity = target - inventory position, where
  target = daily_forecast * (lead_time_days + REVIEW_DAYS) + safety stock.
  Round the quantity up to a whole number of units, then up to a multiple of pack_size,
  then raise it to moq if it is below moq.
- memo.txt, when present, overrides the files.
"""


def make(difficulty, rng):
    n = {1: 4, 2: 6, 3: 8, 4: 10}[difficulty]
    safety_days = rng.choice([2, 3, 4])
    review_days = rng.choice([7, 10, 14])
    today = dt.date(2026, 10, 1) + dt.timedelta(rng.randrange(0, 60))
    skus = [f"SKU-{rng.randrange(100, 999)}" for _ in range(n)]
    while len(set(skus)) < n:
        skus = [f"SKU-{rng.randrange(100, 999)}" for _ in range(n)]
    items = {}
    for s in skus:
        forecast = round(rng.uniform(3, 60), 1)
        lead = rng.randrange(2, 15)
        pack = rng.choice([1, 1, 6, 10, 12, 24]) if difficulty >= 3 else 1
        moq = rng.choice([0, 0, 50, 100, 200]) if difficulty >= 3 else 0
        rop = forecast * lead + math.ceil(round(forecast * safety_days, 6))
        on_hand = max(0, int(rop * rng.uniform(0.3, 1.6)))
        items[s] = {"forecast": forecast, "lead": lead, "pack": pack, "moq": moq, "on_hand": on_hand, "reserved": rng.randrange(0, max(1, on_hand // 4) + 1)}
    pos = []
    if difficulty >= 2:
        for s in rng.sample(skus, max(2, n // 2)):
            eta = today + dt.timedelta(rng.randrange(1, 20))
            status = rng.choice(["confirmed", "confirmed", "confirmed", "cancelled", "draft"])
            pos.append([f"PO-{rng.randrange(1000, 9999)}", s, rng.randrange(20, 400, 10), eta, status])

    notes, discontinued = [], set()
    shown_lead = {s: it["lead"] for s, it in items.items()}  # what suppliers.csv says
    if difficulty == 4:
        s = rng.choice(skus)
        new_lead = items[s]["lead"] + rng.randrange(3, 8)
        notes.append(f"- Supplier change: {s} now has lead_time_days = {new_lead}.")
        items[s]["lead"] = new_lead
        d = rng.choice([x for x in skus if x != s])
        discontinued.add(d)
        notes.append(f"- {d} is discontinued: do not order it.")

    orders = {}
    for s, it in items.items():
        if s in discontinued:
            continue
        incoming = sum(q for _, sku, q, eta, st in pos if sku == s and st == "confirmed" and eta <= today + dt.timedelta(it["lead"]))
        position = it["on_hand"] - it["reserved"] + incoming
        safety = math.ceil(round(it["forecast"] * safety_days, 6))  # round(): 12.3 * 7 is not exactly 86.1
        if position > round(it["forecast"] * it["lead"] + safety, 6):
            continue
        qty = math.ceil(round(it["forecast"] * (it["lead"] + review_days) + safety - position, 6))
        qty = math.ceil(qty / it["pack"]) * it["pack"]
        orders[s] = max(qty, it["moq"])
    if not orders or len(orders) == len(items) - len(discontinued):
        return make(difficulty, rng)  # need a mix of SKUs to order and not to order

    files = {
        "policy.md": POLICY.replace("SAFETY_DAYS", str(safety_days)).replace("REVIEW_DAYS", str(review_days)),
        "today.txt": f"today: {today.isoformat()}\n",
        "stock.csv": csv_text(["sku", "on_hand", "reserved"], [[s, it["on_hand"], it["reserved"]] for s, it in items.items()]),
        "demand.csv": csv_text(["sku", "daily_forecast"], [[s, it["forecast"]] for s, it in items.items()]),
        "suppliers.csv": csv_text(["sku", "lead_time_days", "pack_size", "moq"], [[s, shown_lead[s], it["pack"], it["moq"]] for s, it in items.items()]),
    }
    if pos:
        files["open_pos.csv"] = csv_text(["po_id", "sku", "qty", "eta", "status"], [[p, s, q, e.isoformat(), st] for p, s, q, e, st in pos])
    if notes:
        files["memo.txt"] = "# Purchasing memo\n" + "\n".join(notes) + "\n- Quarterly stocktake is next Friday.\n"
    ws = Workspace(files)

    def grade(answer):
        got = answer.get("orders") if isinstance(answer, dict) else None
        if not isinstance(got, dict):
            return 0.0, 'answer needs "orders": {"SKU-123": quantity, ...}'
        try:
            got = {str(k).strip().upper(): int(round(float(v))) for k, v in got.items() if float(v) > 0}
        except (TypeError, ValueError):
            return 0.0, "quantities must be numbers"
        return (1.0, "correct") if got == orders else (0.0, f"{sum(got.get(k) == v for k, v in orders.items())}/{len(orders)} orders right; extra or wrong orders count against you")

    return Episode(
        question="Decide which SKUs to order today and how many units, following policy.md.",
        tools=file_tools(ws, files),
        answer_hint='Answer: an "orders" object mapping each SKU to order to its quantity (leave out SKUs not to order).\n'
        + submit_hint({"orders": {"SKU-123": 240}}),
        grade=grade,
        workspace=ws,
        oracle=[submit_action({"orders": orders})],
    )


FAMILY = Family("inventory_reorder", "industrial_physical_systems", "Reorder decisions from stock, forecasts, lead times, open POs, pack sizes, MOQs and memos", make)
