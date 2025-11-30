from __future__ import annotations

from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from core.detections import DetectionRaw, LaneSummary


class Renderer:
    def __init__(self, class_names: Optional[Dict[int, str]] = None) -> None:
        self.class_names = class_names or {}
        self._color_cache: Dict[int, tuple[int, int, int]] = {}

    def render(self, frame: np.ndarray, detections: Iterable[DetectionRaw], summary: Optional[LaneSummary] = None) -> np.ndarray:
        img = frame.copy()
        for det in detections:
            color = self._color_for_class(det.class_id)
            if det.mask is not None:
                mask = det.mask
                if mask.ndim == 2:
                    mask = np.expand_dims(mask, axis=-1)
                alpha = 0.25
                mask_f = (mask.astype(np.float32) / 255.0) * alpha
                img = cv2.addWeighted(img, 1.0, np.full_like(img, color), 0.0, 0, dst=img)
                img = (img * (1 - mask_f) + np.array(color, dtype=np.float32) * mask_f).astype(np.uint8)
            if det.bbox is not None:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{self.class_names.get(det.class_id, det.class_id)} {det.confidence:.2f}"
                cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if summary:
            text = (
                f"Left: {summary.left_offset_dm}dm  Right: {summary.right_offset_dm}dm  "
                f"Quality: {summary.quality}"
            )
            cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return img

    def _color_for_class(self, class_id: int) -> tuple[int, int, int]:
        if class_id not in self._color_cache:
            rng = np.random.default_rng(class_id)
            self._color_cache[class_id] = tuple(int(x) for x in rng.integers(0, 255, size=3))
        return self._color_cache[class_id]
