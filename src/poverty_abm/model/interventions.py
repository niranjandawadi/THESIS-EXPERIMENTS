from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

def apply_cash_transfer(wealth: np.ndarray, targets: List[int], budget_total: float) -> None:
    if not targets:
        return
    delta = float(budget_total) / float(len(targets))
    wealth[targets] += delta