"""Base interfaces for reverb engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class ReverbEngine(ABC):
    """Abstract interface for all reverb engines."""

    @abstractmethod
    def process(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Process audio and return an array with the same shape."""
        raise NotImplementedError
