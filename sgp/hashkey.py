from cachetools.keys import _HashedTuple, _kwmark  # noqa
from numpy import ndarray
from pandas.core.generic import NDFrame


def hashkey(*args, **kwargs):
    """Return a cache key for the specified hashable arguments, convert non-hashable arguments to hashable ones."""
    hashable_args = []
    for arg in args + sum(sorted(kwargs.items()), _kwmark) if kwargs else args:
        if isinstance(arg, dict):
            hashable_args.append(frozenset(arg.items()))
        elif isinstance(arg, (list, ndarray)):
            hashable_args.append(tuple(map(tuple, arg)))
        elif isinstance(arg, NDFrame):
            hashable_args.append(arg.to_string())
        else:
            assert hash(arg), f"Argument {arg} is not hashable."
            hashable_args.append(arg)
    return _HashedTuple(hashable_args)
