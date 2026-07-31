import numpy as np
import unittest

from helpers.html_utils import build_interactive_html
from helpers.model_utils import forward_amplitude, fit_physics_params


class PhysicsModelTests(unittest.TestCase):
    def test_fit_physics_params_recovers_known_parameters(self):
        freqs = np.linspace(900.0, 1400.0, 80)
        voltage = np.ones_like(freqs) * 1.2
        theta_true = np.array([1e-8, 2.0, 1200.0, 1.2, 0.8])
        background_true = 0.02

        amps = forward_amplitude(freqs, theta_true, voltage, background=background_true)
        params_hat, background_hat = fit_physics_params(freqs, amps, voltage)

        self.assertTrue(np.isfinite(background_hat))
        self.assertTrue(np.all(np.array(params_hat) > 0))

        pred = forward_amplitude(freqs, params_hat, voltage, background=background_hat)
        self.assertLess(np.mean(np.abs(pred - amps)), 1e-3)
        self.assertLess(abs(background_hat - background_true) / max(background_true, 1e-9), 0.25)

    def test_build_interactive_html_renders_physics_params(self):
        html = build_interactive_html(
            title="Test",
            subtitle="Test subtitle",
            all_data={
                "3": {
                    "train": [1, 2],
                    "freqs": [100.0, 200.0],
                    "actual": [0.1, 0.2],
                    "fwhm_lo": 100.0,
                    "fwhm_hi": 200.0,
                    "Vector": {
                        "Physics": {
                            "pred": [0.1, 0.2],
                            "mae_full": 0.0,
                            "r2_full": 1.0,
                            "mae_roi": 0.0,
                            "r2_roi": 1.0,
                            "params": {"m_N": 1.0, "c_N": 2.0, "k_N": 3.0, "alpha_N": 4.0, "G_N": 5.0, "A_background": 6.0},
                        }
                    },
                    "Pointwise": {},
                    "score_rows": [],
                }
            },
        )

        self.assertIn("<details class=\"score-sec\">", html)
        self.assertIn("Physics Parameters", html)
        self.assertIn("m_N:", html)
        self.assertIn("A_background:", html)


if __name__ == "__main__":
    unittest.main()
