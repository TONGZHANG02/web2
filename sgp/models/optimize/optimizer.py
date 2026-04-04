from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple, Type

import numpy as np
from typing_extensions import Callable

from sgp.pydantic import HashableBaseModelIO


class OptimizerOptions(HashableBaseModelIO):
    """
    Options of the optimizer
    """


class OptimizerResult(HashableBaseModelIO):
    """
    OptimizeResult for the optimizers
    """

    #: The solution of the optimization.
    x: List[float]
    #: Values of objective function
    y: float


class OptimizerBase:
    """
    Base class for optimizers
    """

    #: number of objectives
    n_obj: int = 1

    #: Objective function to optimize, the function's signature must be: f(x: Iterable) -> float
    objective_function: Callable

    #: Initial parameters
    x0: np.ndarray

    #: Parameter bounds
    bounds: np.ndarray | None

    maxiter: int
    maxfun: int

    #: Additional arguments for the objective function
    args: tuple

    #: options
    options: dict = {}

    #: callback function that will be evaluated after each iteration
    user_callback: Callable[[Iterable], Any] | None

    def __init__(
        self,
        objective_function: Callable,
        x0: np.ndarray,
        *,
        args: tuple = (),
        bounds: np.ndarray | None = None,
        callback: Callable[[Iterable], Any] | None = None,
        maxiter: int = 1000,
        maxfun: int = 10000,
        **options,
    ):
        """Initialize the optimizer

        Parameters
        ----------
        objective_function : Callable
            Objective function
        x0 : np.ndarray
            Initial parameters
        args : tuple
            Additional arguments for the objective function
        bounds : np.ndarray
            Bounds of the parameter
        maxiter : int
            Maximal number of iterations
        maxfun : int
            Maximal number of function evaluations
        callback : Callable
            Callback function that will be evaluated after each iteration
        options : int | float | bool | dict | Any
            Optimizer-specific options
        """
        self.objective_function = objective_function
        self.x0 = x0
        self.bounds = bounds
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.user_callback = callback
        self.args = args
        self.options.update(options)
        self.setup()

    def setup(self):
        """Post set up the optimizer"""
        pass

    def callback(self, x: np.ndarray | Any):
        """Callback function that will be evaluated after each iteration"""
        self.user_callback(x) if self.user_callback else None

    def optimize(self) -> OptimizerResult:
        """Optimize the objective function"""
        raise NotImplementedError

    @classmethod
    def fmin(cls, *args, **kwargs) -> OptimizerResult:
        """Static method to do the optimization

        Parameters
        ----------
        args, kwargs
            Positional and keyword arguments for the optimizer

        Returns
        -------
        res : OptimizeResult
            Optimize result
        """
        return cls(*args, **kwargs).optimize()


class Registry(Dict[str, Type[OptimizerBase]]):
    """Registry of all available models"""

    def getName(self, model) -> str:
        """Get the name of the ssi object

        Parameters
        ----------
        model : OptimizerBase
            The model object

        Returns
        -------
        str
            Name of the ssi object
        """
        for name, cls in self.items():
            if isinstance(model, cls):
                return name
        raise ValueError(f"Model {model} is not registered")


registry: Registry[str, Type[OptimizerBase]] = Registry()


def register(
    cls_or_name: Type[OptimizerBase] | str | None = None,
    name: str | None = None,
    *,
    jitclass: bool = False,
    spec: List[Tuple[str, type]] | Mapping[str, Type] | None = None,
    saveto: Registry | None = None,
    **methods,
) -> Type | Callable:
    """Register a new model

    Examples
    --------
    The following calls are equivalent:

    1) ``cls_or_name = "Foo"``, ``name = None``

       >>> @register("Foo")
       ... class Foo:
       ...     ...

    2) ``cls_or_name = None``, ``name = "Foo"``

       >>> @register(name="Foo")
       ... class Foo:
       ...     ...

    3) ``cls_or_name = Foo``, ``name = "Foo"``

       >>> class Foo:
       ...     ...
       >>> register(Foo, "Foo")  # noqa
       <class 'sgp.models.optimize.optimizer.Foo'>

    Parameters
    ----------
    cls_or_name : Type | str, optional
        The class to register or the name of the class, by default None.
    name : str, optional
        The name of the class, by default None
    jitclass : bool, optional
        Whether to compile the class with numba jitclass, by default False
    spec : List[Tuple[str, type]] | Mapping[str, type], optional
        The numba jitclass spec, by default None
    saveto : Registry, optional
        The registry to save the class, by default None which will use the default registry
    methods
        The additional methods to add to the class before the jitclass compilation, functions with the first argument
        being the class instance ``self``

    Returns
    -------
    Type | Callable
        The class that was registered or a callable that registers the class
    """
    saveto_registry = saveto if saveto is not None else registry
    cls: Type[OptimizerBase] | None = cls_or_name
    if cls is not None and not isinstance(cls, type) and name is None:
        assert isinstance(cls, str), "name of the model must be a string"
        cls, name = None, cls

    assert name is not None, "name of the model is required"
    if cls is not None:
        for method, func in methods.items():
            setattr(cls, method, func)

        if jitclass:
            import numba

            saveto_registry[name] = numba.experimental.jitclass(cls, spec)
        else:
            saveto_registry[name] = cls
        return saveto_registry[name]

    return lambda x: register(x, name, jitclass=jitclass, spec=spec, saveto=saveto_registry, **methods)
