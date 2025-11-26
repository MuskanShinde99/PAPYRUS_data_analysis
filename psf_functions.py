#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:21:37 2025

@author: muskanshinde
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Circle
from scipy.io import loadmat
from scipy.ndimage import center_of_mass, fourier_shift
from scipy.optimize import curve_fit

def structured_bkg(img):
    """Compute a synthetic structured background"""
    a_bg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a_bg[i,j] = np.median(np.concatenate((img[i,:],img[:,j])))
    return a_bg

def compute_modes_rms(modes):
    #modes = fits.getdata(os.path.join(folder, 'modes.fits'))
    return np.sqrt(np.mean(modes**2, axis=1))  # (Nt,)\
        
def compute_modes_rms_kl(modes):
    #modes = fits.getdata(os.path.join(folder, 'modes.fits'))
    return np.sqrt(np.mean(modes**2, axis=0))  # (Nt,)


def center_psf(img, nx=None, low=0.1, integer=False):
    """
    Center a PSF using its center of gravity (no background subtraction).
    Optionally crop/pad to (nx,nx).
    """
    y = np.arange(img.shape[0])
    x = np.arange(img.shape[1])
    X, Y = np.meshgrid(x, y)

    # threshold to avoid noisy halo biasing CoG
    img_filt = img - np.median(img)
    img_filt = np.clip(img_filt - low*np.max(img_filt), 0, None)

    cx = np.sum(img_filt*X)/np.sum(img_filt)
    cy = np.sum(img_filt*Y)/np.sum(img_filt)
    if integer:
        cx, cy = int(np.round(cx)), int(np.round(cy))

    # desired center
    cx_target = (img.shape[1]-1)/2
    cy_target = (img.shape[0]-1)/2
    shift = [cy_target-cy, cx_target-cx]

    # subpixel shift
    img_centered = np.fft.ifftn(fourier_shift(np.fft.fftn(img), shift)).real

    # crop/pad if requested
    if nx is not None:
        y0 = (img_centered.shape[0]-nx)//2
        x0 = (img_centered.shape[1]-nx)//2
        img_centered = img_centered[y0:y0+nx, x0:x0+nx]

    return img_centered, (cx, cy)

def compute_integrated_psf(folder, background_subtraction=True, centering=False,
                           nx=150, low=0.1):
    """
    Load PSF cube, subtract background if requested,
    optionally center PSF on its CoG and crop to (nx,nx).
    If centering=False, use fixed crop [225:375, 50:200].
    """
    psf = fits.getdata(os.path.join(folder, 'cblue.fits'))  # shape (nt, ny, nx)

    if background_subtraction:
        # Estimate background from first 100 columns
        background = np.mean(np.mean(psf[:, :100, :], axis=0), axis=0)
        psf = psf - background[None, None, :]

    # Collapse cube into a single PSF
    psf_sum = np.sum(psf, axis=0)

    if centering:
        # use CoG-based centering
        psf_sum, (cx, cy) = center_psf(psf_sum, nx=nx, low=low)
    else:
        # fixed crop (225:375, 50:200), same as your original code
        # psf_sum = psf_sum[225:375, 50:200]
        psf_sum = psf_sum[175:425, 0:250]

    return psf_sum



# ---------- Helper function ----------
def psf_centroid_local(psf, halfwidth=10):
    """Find centroid around brightest pixel to avoid halo bias."""
    psf = np.nan_to_num(psf, nan=0.0)
    
    # Brightest pixel
    y_peak, x_peak = np.unravel_index(np.argmax(psf), psf.shape)
    
    # Crop a window around the peak
    y_min = max(0, y_peak - halfwidth); y_max = min(psf.shape[0], y_peak + halfwidth)
    x_min = max(0, x_peak - halfwidth); x_max = min(psf.shape[1], x_peak + halfwidth)
    crop = psf[y_min:y_max, x_min:x_max]
    
    # Weighted centroid inside the crop
    Y, X = np.indices(crop.shape)
    x_c = (X * crop).sum() / crop.sum() + x_min
    y_c = (Y * crop).sum() / crop.sum() + y_min
    return x_c, y_c


# ---------- Encircled energy function ----------
def encircled_energy(psf, x_c, y_c, radius=20):
    """Compute flux inside a circular aperture around (x_c, y_c)."""
    Y, X = np.indices(psf.shape)
    r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
    mask = r <= radius
    return psf[mask].sum()

def encircled_energy_in_3x3(psf, x_c, y_c, half_size=1):
    """
    Compute flux inside a (2*half_size+1) × (2*half_size+1) square 
    centered on (x_c, y_c).
    
    Parameters
    ----------
    psf : 2D array
        PSF image
    x_c, y_c : float
        Center coordinates (can be float, will be rounded to nearest pixel)
    half_size : int, optional
        Half-size of the box (default=1 → 3×3 region)

    Returns
    -------
    flux : float
        Total flux in the box region
    """
    x_c = int(round(x_c))
    y_c = int(round(y_c))

    y0 = max(0, y_c - half_size)
    y1 = min(psf.shape[0], y_c + half_size + 1)
    x0 = max(0, x_c - half_size)
    x1 = min(psf.shape[1], x_c + half_size + 1)

    return psf[y0:y1, x0:x1].sum()

def gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    """2D axis-aligned Gaussian."""
    x, y = xy
    g = amp * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2) +
          ((y - y0)**2) / (2 * sigma_y**2))
    ) + offset
    return g.ravel()


def fit_fwhm_2d(psf, pixel_scale=None):
    """
    Robust 2D Gaussian fit: returns (FWHM_x, FWHM_y, FWHM_geom), popt, model
    with proper bounds to ensure meaningful fits.
    """

    ny, nx = psf.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    # ----- Initial guess -----
    amp0 = np.nanmax(psf)
    y0, x0 = np.unravel_index(np.nanargmax(psf), psf.shape)
    sigma0 = min(nx, ny) / 6       # more stable initial guess
    offset0 = np.nanmedian(psf)

    p0 = [amp0, x0, y0, sigma0, sigma0, offset0]

    # ----- Bounds to avoid invalid fit -----
    # amp ≥ 0
    # sigma_x, sigma_y ≥ small positive number
    # x0, y0 within image
    bounds_lower = [0,        0,        0,     0.3,    0.3,    -np.inf]
    bounds_upper = [np.inf,   nx,       ny,    nx,     ny,     np.inf]

    try:
        popt, _ = curve_fit(
            gaussian2d,
            (X, Y),
            psf.ravel(),
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=20000
        )
    except Exception as e:
        print("⚠️ Gaussian fit failed:", e)
        return ((np.nan, np.nan, np.nan), None, np.zeros_like(psf))

    amp, x0, y0, sigma_x, sigma_y, offset = popt

    # ----- FWHM conversion -----
    fwhm_x = 2.355 * sigma_x
    fwhm_y = 2.355 * sigma_y

    # Safe geometric mean
    if fwhm_x > 0 and fwhm_y > 0:
        fwhm_geom = np.sqrt(fwhm_x * fwhm_y)
    else:
        fwhm_geom = np.nan

    # Convert to arcsec if needed
    if pixel_scale is not None:
        fwhm_x *= pixel_scale
        fwhm_y *= pixel_scale
        fwhm_geom *= pixel_scale

    # ----- Model image -----
    model = gaussian2d((X, Y), *popt).reshape(psf.shape)

    return (fwhm_x, fwhm_y, fwhm_geom), popt, model



from scipy.optimize import least_squares

# ---------------- Axis-aligned Gaussian model ----------------
def gaussian2d_xy(x, y, amp, xc, yc, sx, sy, offset):
    """Axis-aligned elliptical Gaussian + constant background."""
    return amp * np.exp(-0.5 * (((x - xc)/sx)**2 + ((y - yc)/sy)**2)) + offset


def fit_psf_full_single_gauss_norot(psf):
    """
    Fit a single axis-aligned Gaussian to the entire PSF frame.
    Uses log-domain residuals + robust ('soft_l1') loss to balance core & halo.

    Returns:
      fwhms: (fwhm_x, fwhm_y, fwhm_geom) in pixels
      params_vec: np.array([amp, xc, yc, sx, sy, offset])
      model_image: best-fit model over the full frame
    """
    img = np.clip(psf.astype(float), 0, None)
    ny, nx = img.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # ---- initial guesses from bright-pixel moments ----
    bg0 = np.quantile(img, 0.05)
    thr = np.quantile(img, 0.95)
    m = img >= thr
    if m.sum() < 50:
        m = img > (bg0 + 3*np.std(img))

    w = img[m]; Xm, Ym = X[m], Y[m]
    tot = w.sum() + 1e-12
    xci = (Xm*w).sum()/tot
    yci = (Ym*w).sum()/tot
    varx = (w*((Xm - xci)**2)).sum()/tot
    vary = (w*((Ym - yci)**2)).sum()/tot
    sx_i = np.sqrt(max(varx, 1e-6))
    sy_i = np.sqrt(max(vary, 1e-6))
    amp_i = max(img.max() - bg0, 1e-6)

    p0 = np.array([amp_i, xci, yci, sx_i, sy_i, max(0.0, bg0)])
    lower = np.array([0.0,     0.0,   0.0,  0.5,  0.5,  0.0])
    upper = np.array([np.inf, nx-1, ny-1, 200., 200., np.inf])

    # ---- robust fit in log domain ----
    eps = 1e-12
    data_log = np.log10(np.clip(img, eps, None))

    def resid(p):
        amp, xc, yc, sx, sy, off = p
        model = gaussian2d_xy(X, Y, amp, xc, yc, sx, sy, off)
        return (np.log10(np.clip(model, eps, None)) - data_log).ravel()

    sol = least_squares(
        resid, p0, bounds=(lower, upper),
        loss='soft_l1', f_scale=1.0, max_nfev=40000
    )
    amp, xc, yc, sx, sy, off = sol.x
    model = gaussian2d_xy(X, Y, amp, xc, yc, sx, sy, off)

    # FWHM (pixels)
    fwhm_x = 2.355 * sx
    fwhm_y = 2.355 * sy
    fwhm_geom = np.sqrt(fwhm_x * fwhm_y)

    return (fwhm_x, fwhm_y, fwhm_geom), sol.x, model



def compute_psd(data, fs):
    """
    Compute one-sided Power Spectral Density (PSD) using FFT.

    Parameters
    ----------
    data : array_like
        Input time-series data.
    fs : float
        Sampling frequency.

    Returns
    -------
    freqs : ndarray
        One-sided frequency axis.
    psd : ndarray
        One-sided PSD values.
    """
    N = len(data)
    data_fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Full two-sided PSD
    Pxx_fft = (1 / (fs * N)) * np.abs(data_fft)**2

    # One-sided
    idx = freqs >= 0
    freqs_one = freqs[idx]
    psd_one = 2 * Pxx_fft[idx]

    return freqs_one, psd_one

def smooth_boxcar(signal, N=5):
    kernel = np.ones(N) / N
    return np.convolve(signal, kernel, mode='same')
