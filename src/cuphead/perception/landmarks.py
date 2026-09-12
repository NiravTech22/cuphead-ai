"""Small, explicitly calibrated image patches verify semantic landmarks/HUD.

Reference patches can be a menu title, an overworld sign, an HP label, or the
victory end card. ROI coordinates are fractions of the game drawable; the
encoder still receives the full image. Matching never infers a win from time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .latent_encoder import _as_pil_image


@dataclass(frozen=True)
class Scene:
    label: str
    mode: str
    terminal: bool = False
    hp: int | None = None
    won: bool = False


class LandmarkVerifier:
    def __init__(self, manifest: str | Path):
        import numpy as np
        from PIL import Image

        path = Path(manifest)
        self.document = json.loads(path.read_text())
        self.references = {}
        self._patches = []
        labels = set()
        for node in self.document["landmarks"]:
            if node["label"] in labels:
                raise ValueError("duplicate landmark label")
            labels.add(node["label"])
            mode = node.get("mode", "menu")
            if mode not in {"menu", "overworld", "combat", "death", "victory"}:
                raise ValueError("invalid landmark mode")
            roi = tuple(node.get("roi", (0, 0, 1, 1)))
            if (
                len(roi) != 4
                or not 0 <= roi[0] < roi[2] <= 1
                or not 0 <= roi[1] < roi[3] <= 1
            ):
                raise ValueError("ROI must be normalized left/top/right/bottom")
            threshold = float(node.get("threshold", 0.06))
            if not 0 < threshold < 1:
                raise ValueError("patch threshold must be between 0 and 1")
            image = Image.open(path.parent / node["image"]).convert("RGB")
            patch = self._patch(image, roi)
            if float(np.std(patch)) < 0.01:
                raise ValueError("blank reference patches cannot verify a landmark")
            scene = Scene(
                node["label"],
                mode,
                mode in {"death", "victory"},
                node.get("hp"),
                mode == "victory",
            )
            checks = []
            for check in node.get("checks", []):
                check_roi = tuple(check["roi"])
                check_threshold = float(check.get("threshold", threshold))
                if (
                    len(check_roi) != 4
                    or not 0 <= check_roi[0] < check_roi[2] <= 1
                    or not 0 <= check_roi[1] < check_roi[3] <= 1
                    or not 0 < check_threshold < 1
                ):
                    raise ValueError("invalid verification patch")
                checks.append(
                    (check_roi, check_threshold, self._patch(image, check_roi))
                )
            self._patches.append((scene, roi, threshold, patch, checks))
            self.references[scene.label] = image
        if not self._patches:
            raise ValueError("at least one calibrated landmark required")
        self.margin = float(self.document.get("match_margin", 0.015))
        if not 0 <= self.margin < 1:
            raise ValueError("invalid ambiguity margin")

    @staticmethod
    def _patch(image, roi):
        import numpy as np
        from PIL import Image

        w, h = image.size
        cropped = image.crop(
            (round(roi[0] * w), round(roi[1] * h), round(roi[2] * w), round(roi[3] * h))
        )
        return (
            np.asarray(
                cropped.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32
            )
            / 255
        )

    def classify(self, frame) -> Scene | None:
        import numpy as np

        image = _as_pil_image(frame)
        scores = []
        for scene, roi, threshold, patch, checks in self._patches:
            if any(
                float(np.abs(self._patch(image, cr) - cp).mean()) > ct
                for cr, ct, cp in checks
            ):
                continue
            error = float(np.abs(self._patch(image, roi) - patch).mean())
            scores.append((error, scene, threshold))
        if not scores:
            return None
        scores.sort(key=lambda entry: entry[0])
        best, scene, threshold = scores[0]
        if best > threshold or (len(scores) > 1 and scores[1][0] - best < self.margin):
            return None
        return scene
