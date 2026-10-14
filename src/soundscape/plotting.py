"""Diagnostic figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from soundscape.config import DetectorConfig


def plot_detections(
    scored: pl.DataFrame, config: DetectorConfig, title: str, destination: Path
) -> None:
    """Candidate energies over time, with strikes highlighted.

    Use this to check detector settings: strikes should form a continuous band
    well above the noise floor, and no strike-like candidates should be grey.
    """
    t = scored["time"].to_numpy()
    energy_db = 10 * np.log10(scored["energy"].to_numpy())
    noise_db = 10 * np.log10(scored["noise"].to_numpy())
    strike = scored["is_strike"].to_numpy()
    # Passed the energy criteria but failed the neighbor check.
    isolated = (
        ~strike
        & (scored["cf"].to_numpy() >= config.threshold)
        & (np.nan_to_num(scored["snr_db"].to_numpy()) >= config.min_snr_db)
    )
    other = ~strike & ~isolated

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, height_ratios=[3, 1]
    )
    ax0.scatter(t[other], energy_db[other], s=2, c="0.7", label="candidate")
    if isolated.any():
        ax0.scatter(
            t[isolated], energy_db[isolated], s=4, c="C1", label="isolated (rejected)"
        )
    ax0.scatter(t[strike], energy_db[strike], s=2, c="C3", label="strike")
    ax0.plot(t, noise_db, c="k", lw=0.8, label="noise floor")
    ax0.plot(
        t,
        noise_db + config.min_snr_db,
        c="k",
        lw=0.8,
        ls="--",
        label=f"noise + {config.min_snr_db:g} dB",
    )
    ax0.set_ylabel("Band energy (dB re 1 uPa²)")
    ax0.set_title(
        f"{title}: {strike.sum()} strikes, band {config.band[0]:g}-"
        f"{config.band[1]:g} Hz, threshold {config.threshold:g}"
    )
    ax0.legend(loc="upper right", markerscale=4, fontsize="small")

    cf = scored["cf"].to_numpy()
    ax1.scatter(t[other], cf[other], s=2, c="0.7")
    ax1.scatter(t[isolated], cf[isolated], s=4, c="C1")
    ax1.scatter(t[strike], cf[strike], s=2, c="C3")
    ax1.axhline(config.threshold, c="k", lw=0.8, ls="--")
    ax1.set_yscale("log")
    ax1.set_ylabel("Relative energy")
    ax1.set_xlabel("Time (UTC)")
    fig.autofmt_xdate()

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
