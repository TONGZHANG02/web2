from __future__ import annotations

from typing_extensions import Literal

from .optimizer import register, registry
from .scipyoptimizer import LBFGSB, TNC, NelderMead, Powell, ScipyResult


Optimizers = Literal[
    "Scipy-LBFGSB",
    "Scipy-NelderMead",
    "Scipy-Powell",
    "Scipy-TNC",
    "SKO-DE",
    "SKO-GA",
    "SKO-PSO",
    "SKO-SA",
    "SKO-AFSA",
]

__all__ = [
    "LBFGSB",
    "NelderMead",
    "Powell",
    "TNC",
    "ScipyResult",
    "Optimizers",
    "SKOAFSA",
    "SKODE",
    "SKOGA",
    "SKOPSO",
    "SKOSA",
    "SKOAFSAResult",
    "SKODEResult",
    "SKOGAResult",
    "SKOPSOResult",
    "SKOSAResult",
    "register",
    "registry",
]
