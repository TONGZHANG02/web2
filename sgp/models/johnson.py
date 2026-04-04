from __future__ import annotations

import math
from numbers import Number
from typing import Dict, Literal, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import johnsonsb, johnsonsu, kstest
from scipy.stats import lognorm as johnsonsl
from scipy.stats import norm, rv_continuous

from sgp.pydantic import HashableBaseModelIO


class Johnson(HashableBaseModelIO):
    """Johnson distribution model."""

    #: Tolerance for the determination of the type of Johnson distribution.
    tolerance: float = 0.001

    def __call__(self, z: Sequence[float], data: npt.ArrayLike) -> float:
        """Calculate the fitness of the Johnson distribution."""
        return self.fitness(z, data)

    def fitness(self, z: Sequence[float], data: npt.ArrayLike) -> float:
        """Calculate the fitness of the Johnson distribution."""
        return self.kstest_statistic(z, data)

    def kstest_statistic(self, z: Sequence[float], data: npt.ArrayLike) -> float:
        """Calculate the statistic of the Johnson distribution."""
        try:
            type, aX, bX, aY, bY = self.fit(data, z[0])
            X = self.Y2X(data, type, aX, bX, aY, bY)
            return kstest(X[~np.isnan(X)], "norm").statistic
        except (ValueError, AssertionError):
            return 1.0

    def kstest_pvalue(self, z: Sequence[float], data: npt.ArrayLike) -> float:
        """Calculate the p-value of the Johnson distribution."""
        try:
            type, aX, bX, aY, bY = self.fit(data, z[0])
            X = self.Y2X(data, type, aX, bX, aY, bY)
            return kstest(X[~np.isnan(X)], "norm").pvalue
        except (ValueError, AssertionError):
            return 0.0

    def fit(self, data: npt.ArrayLike, z: float) -> Tuple[Literal["su", "sb", "sl"], float, float, float, float]:
        """Fit the Johnson distribution to the data."""
        data = np.asarray(data)

        Pa, Pb, Pc, Pd = norm.cdf(-3 * z), norm.cdf(-1 * z), norm.cdf(1 * z), norm.cdf(3 * z)
        ya, yb = np.percentile(data, Pa * 100), np.percentile(data, Pb * 100)
        yc, yd = np.percentile(data, Pc * 100), np.percentile(data, Pd * 100)

        m, n, p = yd - yc, yb - ya, yc - yb
        assert m > 0 and n > 0 and p > 0, "m, n, and p must be positive"
        D = m * n / pow(p, 2)
        if D - 1 > self.tolerance:
            aX = 2 * z / math.acosh(0.5 * (m / p + n / p))
            bX = aX * math.asinh((n / p - m / p) / (2 * math.sqrt(D - 1)))
            aY = 2 * p * math.sqrt(D - 1) / ((m / p + n / p - 2) * math.sqrt(m / p + n / p + 2))
            bY = (yc + yb) / 2 + (p * (n / p - m / p)) / (2 * (m / p + n / p - 2))
            return "su", aX, bX, aY, bY
        elif D - 1 < -self.tolerance:
            aX = z / math.acosh(0.5 * math.sqrt((1 + p / m) * (1 + p / n)))
            bX = aX * math.asinh((p / n - p / m) * math.sqrt((1 + p / m) * (1 + p / n) - 4) / (2 * (1 / D - 1)))
            aY = p * math.sqrt(((1 + p / m) * (1 + p / n) - 2) ** 2 - 4) / (1 / D - 1)
            bY = (yc + yb) / 2 - aY / 2 + p * (p / n - p / m) / (2 * (1 / D - 1))
            return "sb", aX, bX, aY, bY
        else:
            assert m / p > 1 and m >= 0, "m / p > 1 and m >= 0 must be true"
            aX = 2 * z / math.log(m / p)
            bXs = aX * math.log((m / p - 1) / (p * math.sqrt(m / p)))
            aY = 0.0
            bY = (yc + yb) / 2 - 0.5 * p * (m / p + 1) / (m / p - 1)
            return "sl", aX, bXs, aY, bY

    @classmethod
    def Y2X(
        cls,
        Y: npt.ArrayLike | Number,
        type: Literal["su", "sb", "sl"],
        aX: float,
        bX: float,
        aY: float,
        bY: float,
    ) -> np.ndarray | Number:
        """Normalize the data to the Johnson distribution."""
        Y = np.asarray(Y) if not isinstance(Y, Number) else Y
        if type == "su":
            Yn = (Y - bY) / aY
            return aX * np.log(Yn + np.sqrt(1 + Yn**2)) + bX
        elif type == "sb":
            Yn = (Y - bY) / aY
            # assert np.any(Yn <= 0) or np.any(Yn > 1), "Yn <= 0 or Yn > 1 must be true"  # nan values if not true
            return aX * np.log(Yn / (1 - Yn)) + bX
        elif type == "sl":
            # assert np.any(Y >= bY), "Y >= bY must be true"  # nan values if not true
            return aX * np.log(Y - bY) + bX
        else:
            raise ValueError(f"Unknown type: {type}")

    @classmethod
    def X2Y(
        cls,
        X: npt.ArrayLike | Number,
        type: Literal["su", "sb", "sl"],
        aX: float,
        bX: float,
        aY: float,
        bY: float,
    ) -> np.ndarray | Number:
        """Denormalize the data from the Johnson distribution."""
        X = np.asarray(X) if not isinstance(X, Number) else X
        if type == "su":
            return bY + aY * np.sinh((X - bX) / aX)
        elif type == "sb":
            return bY + aY / (1 + np.exp(-(X - bX) / aX))
        elif type == "sl":
            return bY + np.exp((X - bX) / aX)
        else:
            raise ValueError(f"Unknown type: {type}")

    def normalize(self, Y: pd.DataFrame, zs: Dict[str, float]) -> pd.DataFrame:
        for col in Y.columns:
            Y[col] = self.Y2X(Y[col], *self.fit(Y[col], zs[col]))
        return Y

    @classmethod
    def dist(
        cls,
        type: Literal["su", "sb", "sl"],
        aX: float,
        bX: float,
        aY: float,
        bY: float,
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> Tuple[rv_continuous, Tuple, Dict[str, ...]]:
        """Get the distribution object and distribution parameters."""
        if type == "su":
            a, b, loc, scale = (bX - mu) / np.sqrt(sigma), aX / np.sqrt(sigma), bY, aY
            return johnsonsu, (a, b), dict(loc=loc, scale=scale)
        elif type == "sb":
            a, b, loc, scale = (bX - mu) / np.sqrt(sigma), aX / np.sqrt(sigma), bY, aY
            return johnsonsb, (a, b), dict(loc=loc, scale=scale)
        elif type == "sl":
            s, loc, scale = 1 / aX * sigma, bY, np.exp(-(bX - mu) / aX)
            return johnsonsl, (s,), dict(loc=loc, scale=scale)
        else:
            raise ValueError(f"Unknown type: {type}")
