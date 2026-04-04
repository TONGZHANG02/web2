from __future__ import annotations

import inspect
import time
from typing import Any, ClassVar, Dict, List, Tuple

import cachetools
import numpy as np
import pandas as pd
from scipy.stats import johnsonsb, johnsonsu
from scipy.stats import lognorm as johnsonsl
from scipy.stats import rv_continuous
from typing_extensions import Literal

from sgp.config import config
from sgp.corr import corr_bootstrap
from sgp.decorators import timing
from sgp.hashkey import hashkey
from sgp.pydantic import HashableBaseModelIO

from .base import PredictionBase
from .johnson import Johnson
from .optimize import Optimizers
from .optimize.optimizer import OptimizerResult, registry

distributions = dict(su=johnsonsu, sb=johnsonsb, sl=johnsonsl)


class DistributionResult(HashableBaseModelIO, arbitrary_types_allowed=True):
    """Result of a distribution prediction."""

    #: the distribution object
    rv: rv_continuous
    #: the positional arguments
    args: Tuple[float, ...]
    #: the keyword arguments
    kwargs: Dict[str, float]
    #: the y values
    y: List[float]
    #: the original y values
    y0: List[float]
    #: the pdf values
    pdf: List[float]
    #: the bounds of the 95% confidence interval
    bounds: List[float]
    #: the probability density function of the corresponding bounds
    pdf_bounds: List[float]
    #: the lower bound of the 95% confidence interval
    lb: float
    #: the upper bound of the 95% confidence interval
    ub: float
    #: the mean of the distribution
    mean: float

    @property
    def exclude(self) -> set[int] | set[str] | dict[int, Any] | dict[str, Any] | None:
        return {"distribution"}


class MPDConditionalResult(HashableBaseModelIO):
    #: unconditioned distribution result
    unconditioning: DistributionResult
    #: conditioned distribution result
    conditioning: DistributionResult
    #: conditioned variables
    conditioned_variables: List[str]
    #: conditioned values
    conditioned_values: Dict[str, float]
    #: predicted variables
    predicted_variables: List[str]


class MPDDistributionalResult(HashableBaseModelIO, arbitrary_types_allowed=True):
    #: variable
    variable: str

    #: fitting statistic
    statistic: float
    #: fitting p-value
    pvalue: float

    #: the distribution object
    rv: rv_continuous
    #: the positional arguments
    args: Tuple[float, ...]
    #: the keyword arguments
    kwargs: Dict[str, float]

    #: distribution type
    type: Literal["su", "sb", "sl"]
    aX: float
    bX: float
    aY: float
    bY: float

    #: original data
    Y0: List[float]
    #: converted data for Johnson fitting
    Y: List[float]
    #: normalized data from the Johnson distribution
    X: List[float]

    @property
    def dist(self) -> Tuple[Literal["su", "sb", "sl"], float, float, float, float]:
        """Get the distribution parameters."""
        return self.type, self.aX, self.bX, self.aY, self.bY


class MPDDistributionalResults(HashableBaseModelIO):
    #: distributional results
    dists: Dict[str, MPDDistributionalResult]
    #: correlation coefficients matrix
    corr: List[List[float]]

    @property
    def C(self) -> np.ndarray:
        """Get the correlation matrix."""
        return np.array(self.corr)

    @property
    def Y0(self) -> np.ndarray:
        """Get the original data."""
        return np.array([result.Y0 for result in self.dists.values()]).T

    @property
    def Y(self) -> np.ndarray:
        """Get the converted data for Johnson fitting."""
        return np.array([result.Y for result in self.dists.values()]).T

    @property
    def X(self) -> np.ndarray:
        """Get the normalized data from the Johnson distribution."""
        return np.array([result.X for result in self.dists.values()]).T


class MPD(PredictionBase):
    """Multivariate Distribution (MPD) model."""

    #: Johnson distribution model.
    johnson: Johnson = Johnson()

    #: The cache to use for the model
    caches: ClassVar[Dict[str, cachetools.LRUCache]] = dict(
        optimize=cachetools.LRUCache(maxsize=128),
        corr_bootstrap=cachetools.LRUCache(maxsize=128),
        fit=cachetools.LRUCache(maxsize=128),
        dist=cachetools.LRUCache(maxsize=128),
        predict=cachetools.LRUCache(maxsize=128),
    )

    def model_post_init(self, __context: Any) -> None:
        self.johnson.tolerance = self.io.database.mpd.tolerance

    @timing
    @cachetools.cachedmethod(lambda self: self.caches["optimize"], key=hashkey)
    def optimize(self, df: pd.DataFrame, *, method: Optimizers, **kwargs):
        """Optimize the z-value for the Johnson distribution."""
        np.seterr(divide="ignore", invalid="ignore")

        results: Dict[str, OptimizerResult] = {}  # noqa
        statistics, pvalues = {}, {}
        for col in df.columns:  # type: str
            x0, bounds = np.array([0.43]), np.array([[0.01, 1.0]])
            OptimizerType = registry[method]
            optimizer = OptimizerType(objective_function=self.johnson, x0=x0, bounds=bounds, args=(df[col],), **kwargs)
            assert (f0 := self.johnson(x0, df[col])) < 1, f"Initial guess is not a valid Johnson distribution for {col}"

            # Optimize
            calibration_message = f"""\
            ==========================================================================================={"=" * len(col)}
            Calibrating the z-value of the Johnson distribution for variable {col}, this may take a while...
            ==========================================================================================={"=" * len(col)}
            Method: {method}
            Initials: {x0}
            Bounds: {bounds}
            Initial objective: {f0:.4f}
            Options: {", ".join(f"{key}={value}" for key, value in optimizer.options.items())}
            """
            print(inspect.cleandoc(calibration_message)) if config.verbose else None
            start_time = time.time()
            result = optimizer.optimize()
            end_time = time.time()
            duration = end_time - start_time

            # Post-process the optimization results
            results[col] = result
            statistics[col] = self.johnson.kstest_statistic(result.x, df[col])
            pvalues[col] = self.johnson.kstest_pvalue(result.x, df[col])
            print(
                f"Optimized z-value = {result.x[0]:.4f} for {col}, objective = {result.y:.4f}, "
                f"statistic = {statistics[col]:.4f}, p = {pvalues[col]:.4f}, took {duration:.4f} seconds.\n"
            ) if config.verbose else None  # fmt: skip
        np.seterr(all=None)
        return results, statistics, pvalues

    @timing
    @cachetools.cachedmethod(lambda self: self.caches["corr_bootstrap"], key=hashkey)
    def corr_bootstrap(self, X: np.ndarray) -> np.ndarray:
        """Calculate the correlation matrix."""
        return corr_bootstrap(X, bootstraps=self.io.database.mpd.bootstraps)

    @timing
    @cachetools.cachedmethod(lambda self: self.caches["fit"], key=hashkey)
    def fit(self):
        """Fit the Johnson distribution to the data.

        Returns
        -------
        A tuple containing the distributions, the optimization results, the statistics, and the p-values.
        """
        df_mpd = self.io.database.df_mpd

        # Optimize the z-value for the Johnson distribution
        kwargs = dict(method=self.io.database.mpd.optimizer, **self.io.database.mpd.optimizer_options)
        results, statistics, pvalues = self.optimize(df_mpd, **kwargs)
        dists = {key: self.johnson.fit(df_mpd[key], z=results[key].x[0]) for key in self.io.database.mpd.inputs}

        # Fit the Johnson distribution to the data
        return dists, results, statistics, pvalues

    @timing
    @cachetools.cachedmethod(lambda self: self.caches["dist"], key=hashkey)
    def dist(self) -> MPDDistributionalResults:
        """Get the distributional results."""
        # Fit the Johnson distribution to the data
        dists, results, statistics, pvalues = self.fit()

        # Normalize the data and calculate the correlation matrix
        df, df_mpd = self.io.database.df, self.io.database.df_mpd
        X = self.johnson.normalize(df_mpd.copy(), {key: result.x[0] for key, result in results.items()}).values
        C: np.ndarray = self.corr_bootstrap(X[~np.isnan(X).any(axis=1), :])  # noqa

        results = MPDDistributionalResults(dists={}, corr=C)  # noqa
        for idx, key in enumerate(self.io.database.keys):
            key1 = self.io.database.Y(key)
            rv, args, kwargs = self.johnson.dist(*dists[key1])
            data: dict = dict(variable=key1, statistic=statistics[key1], pvalue=pvalues[key1])
            data.update(rv=rv, args=args, kwargs=kwargs)
            data.update(type=dists[key1][0], aX=dists[key1][1], bX=dists[key1][2], aY=dists[key1][3], bY=dists[key1][4])
            data.update(Y0=df[key], Y=df_mpd[key1], X=X[:, idx])
            results.dists[key] = MPDDistributionalResult(**data)
        return results

    @timing
    @cachetools.cachedmethod(lambda self: self.caches["predict"], key=hashkey)
    def predict(self, **inputs: float) -> Dict[str, MPDConditionalResult]:
        dist = self.dist()
        dist_dists = {self.io.database.Y(key): value for key, value in dist.dists.items()}
        C, X = dist.C, dist.X
        if inputs:
            # Convert the input values to the multivariate distribution variables
            Ys = self.io.database.Ys(**inputs)

            # Get the conditional and predictive keys and values
            cond_keys, pred_keys = list(Ys.keys()), list(set(self.io.database.mpd.inputs) - set(Ys.keys()))
            X2 = np.array([self.johnson.Y2X(Ys[key], *dist_dists[key].dist) for key in cond_keys])
            cond = np.array([self.io.database.mpd.inputs.index(key) for key in cond_keys])
            pred = np.array([self.io.database.mpd.inputs.index(key) for key in pred_keys])
            Mu1, Mu2 = np.nanmean(X[:, pred], axis=0), np.nanmean(X[:, cond], axis=0)

            # Calculate the conditional mean and covariance
            C11, C12, C21, C22 = C[pred][:, pred], C[pred][:, cond], C[cond][:, pred], C[cond][:, cond]
            Mu = Mu1 + C12 @ np.linalg.inv(C22) @ (X2 - Mu2)
            Cu = C11 - C12 @ np.linalg.inv(C22) @ C21
        else:
            cond_keys, pred_keys = [], self.io.database.mpd.inputs
            Mu, Cu = np.nanmean(X, axis=0), C
            Mu1, C11 = Mu, Cu

        # Calculate the result distributions before and after conditioning
        results = {}
        for idx, key1 in enumerate(pred_keys):
            result = ()
            for mu, sigma in [(Mu1[idx], C11[idx, idx]), (Mu[idx], Cu[idx, idx])]:  # before and after conditioning
                # Generate the x values
                rv, args, kwargs = self.johnson.dist(*dist_dists[key1].dist, mu=mu, sigma=sigma)
                y = np.linspace(rv.ppf(0.001, *args, **kwargs), rv.ppf(0.999, *args, **kwargs), 1000)
                y0 = self.io.database.Xs(**{key1: y})
                key = self.io.database.X(key1)

                # Calculate the bounds of the 95% confidence interval
                bounds = np.linspace(rv.ppf(0.025, *args, **kwargs), rv.ppf(0.975, *args, **kwargs), 1000)
                bounds0 = self.io.database.Xs(**{key1: bounds})
                lb, mean, ub = self.io.database.Xs(**{key1: rv.ppf([0.025, 0.5, 0.975], *args, **kwargs)})[key]

                # Calculate the probability density function
                pdf = rv.pdf(y, *args, **kwargs)
                pdf_bounds = rv.pdf(bounds, *args, **kwargs)

                # Append the result
                data: dict = dict(rv=rv, args=args, kwargs=kwargs)
                data.update(y=y, y0=y0[key], pdf=pdf, bounds=bounds0[key], pdf_bounds=pdf_bounds)
                data.update(lb=lb, ub=ub, mean=mean)
                result += (DistributionResult(**data),)

            results[key] = MPDConditionalResult(
                unconditioning=result[0],
                conditioning=result[1],
                conditioned_variables=cond_keys,
                conditioned_values=inputs,
                predicted_variables=pred_keys,
            )
        return results
