from __future__ import annotations

from abc import ABC
from typing import List, Tuple

from sgp.pydantic import HashableBaseModelIO

from .optimizer import OptimizerBase, OptimizerResult, register


class ScipyResult(OptimizerResult):
    """Represents the optimization result for scipy optimize algorithms

    Notes
    -----
    ``OptimizeResult`` may have additional attributes not listed here depending
    on the specific solver being used. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizeResult.keys` method.
    """

    direc: List[List[float]] = []
    final_simplex: Tuple[List[List[float]], List[float]] = ([], [])
    jac: List[float] = []
    message: str = ""
    success: bool = False
    status: int = -1
    nit: int = -1
    njev: int = -1
    nfev: int = -1


class ScipyOptions(HashableBaseModelIO):
    maxiter: int = 100
    maxfun: int = 1000
    maxfev: int = 1000
    disp: int | None = None


class ScipyOptimizer(OptimizerBase, ABC):
    method: str

    def setup(self):
        self.options.update(maxiter=self.maxiter, maxfun=self.maxfun)
        if "maxfun" in self.options and self.method in ["Nelder-Mead", "Powell"]:
            self.options["maxfev"] = self.options["maxfun"]
        elif "maxfev" in self.options and self.method in ["L-BFGS-B", "TNC"]:
            self.options["maxfun"] = self.options["maxfev"]

    def optimize(self) -> ScipyResult:
        from scipy import optimize

        result = optimize.minimize(
            self.objective_function,
            x0=self.x0,
            args=self.args,
            bounds=self.bounds,
            method=self.method,
            callback=self.callback,
            options=self.options,
        )
        result.y = result.fun
        return ScipyResult(**result)


class LBFGSBOptions(ScipyOptions):
    maxcor: int
    ftol: float
    gtol: float
    eps: float
    iprint: int
    maxls: int
    finite_diff_rel_step: List[float]


@register("Scipy-LBFGSB")
class LBFGSB(ScipyOptimizer):
    method = "L-BFGS-B"


class NelderMeadOptions(ScipyOptions):
    maxfev: int
    return_all: bool
    initial_simplex: List[float]
    xatol: float
    fatol: float
    adaptive: bool


@register("Scipy-NelderMead")
class NelderMead(ScipyOptimizer):
    method = "Nelder-Mead"


class PowellOptions(ScipyOptions):
    maxfev: int
    direc: List[float]
    tol: float


@register("Scipy-Powell")
class Powell(ScipyOptimizer):
    method = "Powell"


class TNCOptions(ScipyOptions):
    eps: float
    scale: List[float]
    offset: float
    maxCGit: int
    eta: float
    stepmx: float
    accuracy: float
    minfev: float
    ftol: float
    xtol: float
    gtol: float
    rescale: float
    finite_diff_rel_step: List[float]


@register("Scipy-TNC")
class TNC(ScipyOptimizer):
    method = "TNC"
