from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Dict

from sgp.io import SGPIO
from sgp.pydantic import HashableBaseModelIO


class PredictionBase(HashableBaseModelIO, metaclass=ABCMeta, arbitrary_types_allowed=True):
    """Base class for prediction models."""

    #: The SGPIO object
    io: SGPIO

    @abstractmethod
    def predict(self, **inputs: float) -> Dict[str, float | HashableBaseModelIO]:
        """Predict the output from the input."""
        raise NotImplementedError
