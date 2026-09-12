"""Optional ML tests; opt in with CUPHEAD_TEST_VJEPA=1 after fetching the checkpoint."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is optional")
class ImageBoundaryTests(unittest.TestCase):
    def test_raw_capture_bytes_keep_dimensions_and_rgb_order(self):
        from cuphead.perception.capture import Frame
        from cuphead.perception.latent_encoder import _as_pil_image

        image = _as_pil_image(Frame(0, 0, b"\xff\x00\x00\x00\xff\x00", 1, 2, 1))
        self.assertEqual(image.size, (2, 1))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(image.getpixel((1, 0)), (0, 255, 0))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is optional")
    def test_landmark_reference_cross_matching_and_blank_rejection(self):
        from PIL import Image

        from cuphead.perception.landmarks import LandmarkVerifier

        verifier = LandmarkVerifier(REPO / "data/landmarks/forest_follies.json")
        for label, image in verifier.references.items():
            self.assertEqual(verifier.classify(image).label, label)
        self.assertIsNone(verifier.classify(Image.new("RGB", (960, 540))))


@unittest.skipUnless(
    os.environ.get("CUPHEAD_TEST_VJEPA") == "1", "requires opt-in pretrained checkpoint"
)
class PretrainedEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from PIL import Image, ImageDraw

        from cuphead.perception.frozen_jepa import FrozenJEPAEncoder

        cls.torch = torch
        cls.encoder = FrozenJEPAEncoder(
            repository=REPO / "checkpoints/vjepa2",
            checkpoint=REPO / "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
        )
        cls.left = Image.new("RGB", (960, 540), (60, 120, 200))
        draw = ImageDraw.Draw(cls.left)
        draw.rectangle((100, 200, 200, 400), fill="white")
        cls.right = Image.new("RGB", (960, 540), (60, 120, 200))
        draw = ImageDraw.Draw(cls.right)
        draw.rectangle((600, 200, 700, 400), fill="red")

    def test_pretrained_encoder_is_frozen_quantized_and_deterministic(self):
        enc = self.encoder
        self.assertTrue(all(not p.requires_grad for p in enc._model.parameters()))
        self.assertFalse(enc._model.training)
        self.assertGreater(
            sum(
                isinstance(m, self.torch.ao.nn.quantized.dynamic.Linear)
                for m in enc._model.modules()
            ),
            0,
        )
        a = enc.encode(self.left)
        b = enc.encode(self.left)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 384)
        self.assertAlmostEqual(sum(x * x for x in a), 1, places=5)
        self.assertEqual(enc.last_input_shape, (1, 3, 1, 128, 128))
        self.assertGreater(
            sum((x - y) ** 2 for x, y in zip(a, enc.encode(self.right))), 1e-5
        )

    def test_temporal_clip_is_consecutive_and_uses_high_detail(self):
        enc = self.encoder
        a = enc.encode_clip([self.left] * 4)
        b = enc.encode_clip([self.left, self.left, self.right, self.right])
        self.assertEqual(enc.last_input_shape, (1, 3, 4, 256, 256))
        self.assertEqual(len(a), 384)
        self.assertGreater(sum((x - y) ** 2 for x, y in zip(a, b)), 1e-5)
        enc.reset()
        self.assertEqual(len(enc._history), 0)


if __name__ == "__main__":
    unittest.main()
