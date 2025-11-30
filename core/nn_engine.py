from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from core.detections import DetectionRaw

logger = logging.getLogger(__name__)


class NNEngine:
    """
    Adapter around the NN described in use_NN.md.

    TODO: Populate preprocessing/postprocessing once the exact input size,
    color format, normalization, and output tensors are clarified. Current
    use_NN.md documents training flow but not runtime inference specifics.
    """

    def __init__(self, model_path: str, device: str = "cpu", input_size: tuple[int, int] | None = None):
        self.model_path = Path(model_path)
        self.device = device
        self.input_size = input_size
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.error("Model file not found: %s", self.model_path)
            return
        # Placeholder: replace with actual model loading once API is known.
        logger.info("NNEngine initialized with model at %s (device=%s)", self.model_path, self.device)

    def infer(self, frame: np.ndarray) -> List[DetectionRaw]:
        """
        Run inference on a single RGB frame.

        Returns a list of DetectionRaw objects. Implementation is pending
        until the model's runtime API and output format are defined.
        """
        raise NotImplementedError("NNEngine.infer needs implementation per use_NN.md runtime details")
