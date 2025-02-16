# -*- coding: utf-8 -*-
"""
Constructors for environment models.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""
import numpy as np


def construct_environment() -> dict:
    """Construct a simple environment model.

    Returns:
        Dictionary containing the environment model.
    """
    z = np.array([0, 35])
    return {
        # 1. General
        "title": "SouthFork",
        "model": "KRAKEN",
        # 2. Top medium (halfspace)
        "top_opt": "CVW    ",
        # 3. Layered media
        "layerdata": [
            {"z": z, "c_p": 1500.0, "rho": 1.0},
            {"z": [z.max(), 200], "c_p": 1600.0, "rho": 1.5, "a_p": 0.2},
        ],
        # 4. Bottom medium
        "bot_opt": "A",
        "bot_c_p": 1600.0,
        "bot_rho": 1.5,
        "bot_a_p": 0.2,
        # 5. Speed constraints
        "clow": 0,
        "chigh": 1600,
        # 6. Receiver parameters
        "rec_z": 20.0,
        "rec_r": 2.0,
        # 7. Source parameters
        "src_z": 25.0,
    }
