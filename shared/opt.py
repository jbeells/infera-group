# /shared/opt.py
"""
Lightweight optimization wrappers.

Example: 0-1 Knapsack with binary decisions.

Public API:
- solve_knapsack(items, weights, values, capacity) -> dict[item] = 0/1
"""

from typing import Dict, List
try:
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum
    _PUlp_AVAILABLE = True
except Exception:
    LpProblem = None
    LpMaximize = None
    LpVariable = None
    lpSum = None
    _PUlp_AVAILABLE = False

def solve_knapsack(items, capacity):
    if not _PUlp_AVAILABLE:
        raise RuntimeError("PuLP is not installed. Install 'pulp' to use solve_knapsack.")
    prob = LpProblem("Knapsack", LpMaximize)
    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in items}
    prob += lpSum(values[i] * x[i] for i in items)
    prob += lpSum(weights[i] * x[i] for i in items) <= capacity
    prob.solve()
    return {i: int(x[i].value()) for i in items}
