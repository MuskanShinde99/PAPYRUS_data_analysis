"""Common helpers shared across PAPYRUS analysis notebooks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits


def compute_response_matrix(images: Sequence[np.ndarray], mask: np.ndarray | None = None) -> np.ndarray:
    """Flatten a stack of images into a response matrix.

    Parameters
    ----------
    images:
        3D array (n_images, height, width), list of 2D arrays, or 2D array of
        flattened images. Any list/array-like input is converted to a NumPy
        array internally.
    mask:
        Optional 2D mask. When provided only the pixels where ``mask > 0`` are
        retained for each image.

    Returns
    -------
    np.ndarray
        2D matrix with one flattened (and optionally masked) image per row.
    """

    images = np.asarray(images)

    if images.ndim == 3:
        prepared = images
    elif images.ndim == 2:
        n_images, n_pixels = images.shape
        side = int(math.isqrt(n_pixels))
        if side * side == n_pixels:
            prepared = images.reshape((n_images, side, side))
        else:
            prepared = images.reshape((n_images, 1, n_pixels))
    else:
        raise ValueError("Input must be a 2D or 3D array of images.")

    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != prepared.shape[1:]:
            raise ValueError("Mask shape must match the shape of each image.")
        response_matrix = np.array([img[mask > 0].ravel() for img in prepared])
    else:
        response_matrix = prepared.reshape(prepared.shape[0], -1)

    return response_matrix


def compute_modes_rms(folder: str | Path, filename: str = "modes.fits") -> np.ndarray:
    """Compute the RMS of modal coefficients stored in a FITS cube."""

    modes_path = Path(folder) / filename
    modes = fits.getdata(modes_path)
    return np.sqrt(np.mean(modes**2, axis=1))

