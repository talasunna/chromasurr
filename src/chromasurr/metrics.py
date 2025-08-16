"""Extract chromatographic performance metrics from CADET SimulationResults."""

import numpy as np
from CADETProcess.simulationResults import SimulationResults


def extract(sim: SimulationResults, chromatogram_index: int = 0) -> dict[str, float]:
    """
    Compute FWHM, retention time, and number of theoretical plates.

    Parameters
    ----------
    sim : SimulationResults
        The results from `Cadet().simulate(process)`.
    chromatogram_index : int, optional
        Index of chromatogram to use (default is 0).

    Returns
    -------
    peak_width : float
        Full width at half maximum (FWHM).
    retention_time : float
        Time at peak maximum.
    num_plates : float
        Theoretical number of plates (efficiency).

    Raises
    ------
    ValueError
        If `chromatogram_index` is out of bounds.
    """
    n = len(sim.chromatograms)
    if chromatogram_index < 0 or chromatogram_index >= n:
        raise ValueError(
            f"chromatogram_index must be between 0 and {n-1}, "
            f"but got {chromatogram_index}"
        )

    chrom = sim.chromatograms[chromatogram_index]
    t = np.asarray(chrom.time)
    try:
        c = np.asarray(chrom.total_concentration)
    except AttributeError:
        c = np.asarray(chrom.c)

    idx_max = int(np.argmax(c))
    peak_height = float(c[idx_max].item())
    half_max = peak_height / 2.0

    left_idx = np.where(c[:idx_max] <= half_max)[0]
    left = t[left_idx[-1]] if len(left_idx) > 0 else t[0]

    right_idx = np.where(c[idx_max:] <= half_max)[0]
    right = t[idx_max + right_idx[0]] if len(right_idx) > 0 else t[-1]

    peak_width = float(right - left)
    retention_time = float(t[idx_max])

    if peak_width > 0:
        num_plates = 5.54 * (retention_time / peak_width) ** 2
    else:
        num_plates = float("nan")

    return {
        "peak_width": peak_width,
        "retention_time": retention_time,
        "num_plates": num_plates,
    }
