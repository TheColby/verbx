from abc import ABC, abstractmethod

import numpy as np


class ReverbEngine(ABC):
    """Abstract base class for all reverb engines."""

    @abstractmethod
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Process the input audio with the reverb engine.

        Args:
            audio: Input audio array (numpy.ndarray).
            sr: Sample rate (int).

        Returns:
            Processed audio array (numpy.ndarray).
        """
        ...
