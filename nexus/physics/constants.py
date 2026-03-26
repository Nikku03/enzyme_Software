from __future__ import annotations

ATOMIC_MASSES = {
    1: 1.00784,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998403163,
    15: 30.973761998,
    16: 32.06,
    17: 35.45,
    35: 79.904,
    53: 126.90447,
}

KCAL_PER_MOL_TO_EV = 0.0433641153087705
EV_TO_KCAL_PER_MOL = 23.06054783061903

__all__ = [
    "ATOMIC_MASSES",
    "KCAL_PER_MOL_TO_EV",
    "EV_TO_KCAL_PER_MOL",
]
