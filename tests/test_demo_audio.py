import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class Batch:
    """Minimal batch/channel/sample indexing without loading the ML stack."""
    def __init__(self, samples):
        self.data = [[samples]]

    def __getitem__(self, index):
        batch, channel = index
        return self.data[batch][channel]


class DemoAudioTests(unittest.TestCase):
    def test_rate_shape_duration_and_parameters(self):
        tree = ast.parse((Path(__file__).parents[1] / "app.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "inference")
        for rate in (48000, 24000):
            with self.subTest(rate=rate):
                samples = [0.0, 0.25, -0.25] * (rate // 3)
                model = SimpleNamespace(sampling_rate=rate)
                build = Mock(return_value=model)
                upscale = Mock(return_value=Batch(samples))
                scope = dict(build_model=build, super_resolution=upscale)
                exec(compile(ast.Module(body=[fn], type_ignores=[]), "app.py", "exec"), scope)
                actual_rate, actual_samples = scope["inference"]("synthetic.wav", "speech", 4.0, 20)
                self.assertEqual(actual_rate, rate)
                self.assertIs(actual_samples, samples)
                self.assertEqual(len(actual_samples) / actual_rate, 1.0)
                build.assert_called_once_with(model_name="speech")
                upscale.assert_called_once_with(model, "synthetic.wav", guidance_scale=4.0, ddim_steps=20)


if __name__ == "__main__":
    unittest.main()
