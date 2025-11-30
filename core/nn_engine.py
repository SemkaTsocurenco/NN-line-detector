from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from core.detections import DetectionRaw

logger = logging.getLogger(__name__)


class NNEngine:
    """Adapter around the NN described in use_NN.md (segmentation)."""

    def __init__(self, model_path: str, device: str = "cpu", input_size: tuple[int, int] | None = None):
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.input_size = input_size
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and torch.cuda.is_available():
            return device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to cpu")
        return "cpu"

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.error("Model file not found: %s", self.model_path)
            return
        try:
            self.model = torch.jit.load(self.model_path.as_posix(), map_location=self.device)
            logger.info("Loaded TorchScript model from %s", self.model_path)
        except Exception as exc_script:
            logger.warning("TorchScript load failed (%s), trying torch.load", exc_script)
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.model = checkpoint["model"]
                else:
                    self.model = checkpoint
                logger.info("Loaded model via torch.load from %s", self.model_path)
            except Exception as exc:
                logger.exception("Failed to load model: %s", exc)
                self.model = None
                return
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            logger.info("NNEngine initialized (device=%s)", self.device)

    def infer(self, frame: np.ndarray) -> List[DetectionRaw]:
        """
        Run inference on a single RGB frame.

        Returns a list of DetectionRaw objects extracted from segmentation mask.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        orig_h, orig_w, _ = frame.shape
        input_tensor = self._preprocess(frame)

        with torch.no_grad():
            output = self.model(input_tensor)

        logits = self._extract_logits(output)
        if logits is None:
            raise RuntimeError("Model output not understood")

        probs = F.softmax(logits, dim=1)
        conf_map, class_map = torch.max(probs, dim=1)  # shapes (1, H, W)

        conf_np = conf_map.squeeze(0).cpu().numpy()
        class_np = class_map.squeeze(0).cpu().numpy().astype(np.uint8)

        if (logits.shape[2], logits.shape[3]) != (orig_h, orig_w):
            class_np = cv2.resize(class_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            conf_np = cv2.resize(conf_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        detections: List[DetectionRaw] = []
        for class_id in np.unique(class_np):
            if class_id == 0:
                continue  # background
            mask = (class_np == class_id)
            area = int(mask.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(mask)
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            confidence = float(conf_np[mask].mean())
            detections.append(
                DetectionRaw(
                    class_id=int(class_id),
                    confidence=confidence,
                    mask=(mask.astype(np.uint8) * 255),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return detections

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = frame
        if self.input_size:
            w, h = self.input_size
            img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return tensor

    @staticmethod
    def _extract_logits(output: torch.Tensor | List | dict | tuple) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and output:
            if isinstance(output[0], torch.Tensor):
                return output[0]
            if isinstance(output[0], dict) and "out" in output[0]:
                return output[0]["out"]
        if isinstance(output, dict):
            if "out" in output and isinstance(output["out"], torch.Tensor):
                return output["out"]
        return None
