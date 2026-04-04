from typing import ClassVar, Dict

import cachetools
import sympy as sp

from sgp.decorators import timing
from sgp.hashkey import hashkey

from .base import PredictionBase


class ANN(PredictionBase):
    """Artificial Neural Network (ANN) model."""

    #: The cache to use for the model
    cache: ClassVar[cachetools.LRUCache] = cachetools.LRUCache(maxsize=128)

    @timing
    @cachetools.cachedmethod(lambda self: self.cache, key=hashkey)
    def predict(self, **inputs: float) -> Dict[str, float]:
        results = {}
        for key, equation in self.io.database.ann.equations.items():
            if key in inputs:
                continue
            if all(var in inputs for var in equation.inputs):
                expr = sp.sympify(equation.equation, locals={var: sp.Symbol(var) for var in equation.inputs})
                results[key] = expr.subs({sp.Symbol(var): value for var, value in inputs.items()})
        return results
