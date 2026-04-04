from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

import cachetools
import numpy as np
import pandas as pd
import sympy as sp
from typing_extensions import Literal

from .hashkey import hashkey
from .pydantic import HashableBaseModelIO


class EquationIO(HashableBaseModelIO):

    #: Variables required for the equation.
    inputs: List[str]
    #: Equation in string format.
    equation: str


class ANNIO(HashableBaseModelIO):
    equations: Dict[str, EquationIO]


class VariableIO(HashableBaseModelIO):
    #: Name of the output.
    name: str
    #: Unit of the output.
    unit: str
    #: Number of decimals for the input.
    decimals: int | Literal["scientific"]


class MPDIO(HashableBaseModelIO):
    #: Variables for the multivariate distribution.
    inputs: List[str]

    #: Tolerance for the determination of the type of Johnson distribution.
    tolerance: float = 0.01
    #: Number of bootstraps for the determination of the correlation matrix.
    bootstraps: int = 1000
    #: Optimization method to optimize the z-value.
    optimizer: str = "Scipy-LBFGSB"
    #: Options for the optimizer.
    optimizer_options: Dict[str, Any] = {}

    @property
    def keys(self) -> List[str]:
        """Get the keys for the multivariate distribution."""
        return list(self.inputs)


class HSSParametersIO(HashableBaseModelIO, arbitrary_types_allowed=True):
    #: Input parameters for the HSS model.
    inputs: List[str]
    #: Output parameters for the HSS model.
    outputs: Dict[str, str]

    #: The cache to use for the model
    cache: ClassVar[cachetools.LRUCache] = cachetools.LRUCache(maxsize=128)

    @cachetools.cachedmethod(lambda self: self.cache, key=hashkey)
    def predict(self, inputs: Dict[str, float]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Get the output properties for the HSS model."""
        equations, outputs = {}, {}
        for name, equation in self.outputs.items():
            if all(var in inputs for var in self.inputs):
                expr = sp.sympify(equation, locals={var: sp.Symbol(var) for var in self.inputs})
                symbols = [str(s) for s in expr.free_symbols]
                equations[name] = f"{name}({', '.join(symbols)}) = {expr}" if len(symbols) > 0 else f"{name} = {expr}"
                try:
                    outputs[name] = float(expr.subs({sp.Symbol(var): value for var, value in inputs.items()}))
                except TypeError:
                    outputs[name] = -1.0
        return equations, outputs


class InputIO(VariableIO, arbitrary_types_allowed=True):
    #: Default value for the input.
    default: float
    #: Minimum value for the input.
    minimum: float
    #: Maximum value for the input.
    maximum: float
    #: Number of decimals for the input.
    decimals: int
    #: Step size for the input.
    singleStep: float

    #: The cache to use for the model
    cache: ClassVar[cachetools.LRUCache] = cachetools.LRUCache(maxsize=128)


class SoilParametersIO(HashableBaseModelIO):
    #: Base parameters for the HSS model.
    parameters: List[HSSParametersIO]
    #: Updated parameters for the HSS model.
    updates: List[HSSParametersIO]


class HSSIO(HashableBaseModelIO, arbitrary_types_allowed=True):
    #: Possible inputs for the HSS model.
    inputs: List[InputIO]
    #: Possible outputs for the HSS model.
    outputs: List[VariableIO]

    #: Parameters for the HSS model for clay/sand soils, key for the soil type.
    parameters: Dict[str, SoilParametersIO]

    #: The cache to use for the model
    cache: ClassVar[cachetools.LRUCache] = cachetools.LRUCache(maxsize=128)

    @cachetools.cachedmethod(lambda self: self.cache, key=hashkey)
    def predict(self, soilType: str, inputs: Dict[str, float]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Get the output properties for the HSS model."""
        equations, outputs = {}, {}
        for parameters in self.parameters[soilType].parameters + self.parameters[soilType].updates:
            result = parameters.predict(inputs)
            equations.update(result[0]), outputs.update(result[1])
        return equations, outputs


class DatabaseIO(HashableBaseModelIO):
    #: List of keys for the database.
    inputs: Dict[str, VariableIO]
    #: Data table for the database.
    data: List[List[float | None]] = []
    #: Statistics for the database.
    stats: Dict[str, Dict[str, Any]] = {}

    #: IO for the artificial neural network model.
    ann: ANNIO
    #: IO for the multivariate distribution model.
    mpd: MPDIO
    #: IO for the HSS model parameters.
    hss: HSSIO

    @property
    def keys(self) -> List[str]:
        """Get the keys for the database."""
        return list(self.inputs.keys())

    @property
    def df(self) -> pd.DataFrame:
        """Get the data for the database.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the data for the database.
        """
        return pd.DataFrame(self.data, columns=self.keys)

    @property
    def YX(self) -> Dict[str, Tuple[sp.Expr, List[str]]]:
        """Symbolic expressions for the variables in the multivariate distribution from the original variables.

        Returns
        -------
        Dict[str, Tuple[sp.Expr, List[str]]]
            A dictionary of (symbolic expression, free symbols) for each variable.
        """
        results = {}
        for var in self.mpd.inputs:
            expr = sp.sympify(var, locals={v: sp.Symbol(v) for v in self.keys})
            symbols = [str(s) for s in expr.free_symbols]
            results[var] = (expr, symbols)
        return results

    def Y(self, X: str) -> str:
        """Get the variable in the multivariate distribution from the original variable."""
        expr, symbols = self.XY[X]
        assert len(symbols) == 1, "Only one variable is expected"
        return self.mpd.keys[int(symbols[0][1:]) - 1]

    def Ys(self, **xs: np.ndarray | float) -> Dict[str, np.ndarray | float]:
        """Get the values for the variables in the multivariate distribution from the original variables."""
        results = {}
        for var, (expr, symbols) in self.YX.items():
            if all(s in xs for s in symbols):
                func = sp.lambdify(symbols, expr, "numpy")
                results[var] = func(**{s: xs[s] for s in symbols})
        return results

    @property
    def XY(self) -> Dict[str, Tuple[sp.Expr, List[str]]]:
        """Symbolic expressions for the original variables from the variables in the multivariate distribution.

        Returns
        -------
        Dict[str, Tuple[sp.Expr, List[str]]]
            A dictionary of (symbolic expression, free symbols) for each variable.
        """
        equations = [expr - sp.Symbol(f"Y{i+1}") for i, (expr, _) in enumerate(self.YX.values())]
        results: Tuple[sp.Expr, ...] = sp.solve(equations, [sp.Symbol(key) for key in self.keys])[0]
        return {key: (results[i], [str(s) for s in results[i].free_symbols]) for i, key in enumerate(self.keys)}

    def X(self, Y: str) -> str:
        """Get the original variable from the variable in the multivariate distribution."""
        expr, symbols = self.YX[Y]
        assert len(symbols) == 1, "Only one variable is expected"
        return symbols[0]

    def Xs(self, **ys: np.ndarray | float) -> Dict[str, np.ndarray | float]:
        """Get the values for the original variables from the variables in the multivariate distribution."""
        results = {}
        ys = {f"Y{self.mpd.inputs.index(k) + 1}": v for k, v in ys.items()}
        for var, (expr, symbols) in self.XY.items():
            if all(s in ys for s in symbols):
                func = sp.lambdify(symbols, expr, "numpy")
                results[var] = func(**{s: ys[s] for s in symbols})
        return results

    @property
    def df_mpd(self) -> pd.DataFrame:
        """Get the data for the multivariate distribution.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the data for the multivariate distribution.
        """
        data = {}
        df = self.df
        data_original = {var: df[var] for var in self.keys}
        for var, (expr, symbols) in self.YX.items():
            func = sp.lambdify(symbols, expr, "numpy")
            data[var] = func(**{str(v): data_original[str(v)] for v in expr.free_symbols})
        return pd.DataFrame(data)

    def mean(self, key: str) -> float:
        """Get the mean of the database key."""
        return self.df[key].mean()

    def min(self, key: str) -> float:
        """Get the minimum of the database key."""
        return self.df[key].min()

    def max(self, key: str) -> float:
        """Get the maximum of the database key."""
        return self.df[key].max()

    def std(self, key: str) -> float:
        """Get the standard deviation of the database key."""
        return self.df[key].std()

    def cov(self, key: str) -> float:
        """Get the coefficient of variation of the database key."""
        return self.std(key) / self.mean(key)

    def describe(self, key: str) -> Tuple[float, ...]:
        """Get the statistics of the database key."""
        return self.mean(key), self.min(key), self.max(key), self.std(key), self.cov(key)


class SGPIO(HashableBaseModelIO):
    database: DatabaseIO
