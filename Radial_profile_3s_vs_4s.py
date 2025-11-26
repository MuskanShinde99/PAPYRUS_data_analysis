#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSF fitting and comparison between 3PWFS and 4PWFS
Author: Muskan Shinde
Date: Oct 26, 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle
from astropy.io import fits
from numpy.fft import fft2, fftshift
from maoppy.utils import circavg
from maoppy.instrument import papyrus
from maoppy.psfmodel import Psfao, Turbulent
from maoppy.psffit import psffit
from papylib.image import strehl_ratio, center_cog, compute_cog

# Try importing the polychromatic AO PSF model
try:
    from maoppy.psfmodel import PsfaoPolychromatic
except ImportError:
    print('Warning: PsfaoPolychromatic not found. Using fallback.')
    class PsfaoPolychromatic(Psfao):
        def __init__(self, *args, samp=None, **kwargs):
            super().__init__(*args, samp=np.mean(samp), **kwargs)

#------------------------------------------------------------------------------
# General Parameters
#------------------------------------------------------------------------------
nx = 100                     # Crop size [pix]
nb_mode_control = 195
elevation_deg = 80
pix_mas = 70.6               # Pixel scale [mas/pix]
wvl = 1400e-9                # Central wavelength [m]
TELESCOPE_DIAMETER = 1.52    # m
TELESCOPE_OBSTRUCTION = 0.27
DM_D_CALIB = 37.5            # mm
DM_D_SKY = 35.5              # mm
DM_PITCH = 2.5               # mm
DM_NACT = 241
NCPA_ASTIG_NM = 0
NCPA_TREFOIL_NM = 20
rad2arcsec = 180/np.pi * 3600

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------
def load_fits(path):
    """Load a FITS cube as float array"""
    with fits.open(path) as f:
        return f[0].data.astype(float)

def get_otf(img):
    """Compute Optical Transfer Function magnitude"""
    return np.abs(fftshift(fft2(fftshift(img))))

#------------------------------------------------------------------------------
# Process one PSF (used for both 3s and 4s)
#------------------------------------------------------------------------------
def process_psf(psf_path):
    cube = load_fits(psf_path)
    img = np.var(cube, axis=0)[1:-1, 1:-1]
    #bkg = np.zeros_like(img)

    cx, cy = compute_cog(img, bkg, integer=True)
    print(f"COG center: ({cx:.1f}, {cy:.1f})")

    cube_c = cube[:-2, cy-nx//2:cy+nx//2, cx-nx//2:cx+nx//2]
    psf_c, bkg_c = center_cog(img, bkg, nx)
    #psf_c -= bkg_c

    # Sampling and AO setup
    sampling = rad2arcsec * wvl / TELESCOPE_DIAMETER / (pix_mas * 1e-3)
    papyrus.occ = TELESCOPE_OBSTRUCTION
    nb_act_lin = 1 + DM_D_CALIB / DM_PITCH
    papyrus.Nact = round(nb_act_lin * DM_D_SKY / DM_D_CALIB * np.sqrt(nb_mode_control / DM_NACT))

    samp_list = np.array([1, 1.0/1.4, 1.1/1.4, 1.2/1.4, 1.3/1.4]) * sampling

    # Build PSF model
    psfmodel = PsfaoPolychromatic((nx, nx), system=papyrus, samp=samp_list)
    psfparam_guess = [0.09, 1e-4, 0.4, 0.5, 1, 0, 1.5]
    fixed = [False] * 7
    psfmodel.zernike = np.array([0, 0, NCPA_ASTIG_NM, 0, 0, 0, NCPA_TREFOIL_NM]) * 2 * np.pi / (wvl * 1e9)

    # Fit PSF
    out = psffit(psf_c, psfmodel, psfparam_guess, fixed=fixed, max_nfev=30)

    # Compute normalized PSF and Strehl
    psf_norm = psf_c/np.sum(psf_c)#(psf_c - out.flux_bck[1]) / out.flux_bck[0]
    sr = strehl_ratio(psf_norm, sampling)
    sr_fit = np.array(psfmodel.strehlOTF(out.x))
    print(f"Strehl (OTF): {100*sr:.1f} % | (Fit): {100*sr_fit:.1f} %")

    # Compute seeing
    r0_zenith = out.x[0] / np.cos(np.pi/2 - elevation_deg*np.pi/180)**(3/5)
    seeing = rad2arcsec * wvl / r0_zenith
    seeing_550 = seeing * (550e-9 / wvl)**(-1/5)
    print(f"Seeing: {seeing_550:.2f}\" (zenith @ 550 nm)")

    return psf_norm, out, psfmodel, sampling

#------------------------------------------------------------------------------
# Paths to data
#------------------------------------------------------------------------------
path_3s = '/Volumes/SanDisk/PAPYRUS_run2/01.10.25/record/2025-10-02_03-27-19/cred2.fits'
path_4s = '/Volumes/SanDisk/PAPYRUS_run2/01.10.25/record/2025-10-02_03-30-43/cred2.fits'

#------------------------------------------------------------------------------
# Process 3PWFS and 4PWFS PSFs
#------------------------------------------------------------------------------
print("\n--- Processing 3PWFS ---")
psf_norm_3s, out_3s, psfmodel_3s, sampling_3s = process_psf(path_3s)

print("\n--- Processing 4PWFS ---")
psf_norm_4s, out_4s, psfmodel_4s, sampling_4s = process_psf(path_4s)

#%%
#------------------------------------------------------------------------------
# Compare PSFs (Radial Profiles)
#------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.title('PSF Radial Profiles', fontsize=18, weight='bold')

# Centers in cropped coordinates + subpixel correction
center_diffrac = (nx // 2, nx // 2)
center_3s = (nx // 2 + out_3s.dxdy[1], nx // 2 + out_3s.dxdy[0])
center_4s = (nx // 2 + out_4s.dxdy[1], nx // 2 + out_4s.dxdy[0])

# ✅ Normalize with diffraction-limited PSF peak
maxi = psf_norm_3s.max()

# --- Custom dark colors ---
colors = {
    '3s': 'yellow',  # dark purple (indigo)
    '4s': '#4B0082'   # rich dark blue
}

# Plot PSF radial profiles
plt.semilogy(
    circavg(psf_norm_3s / maxi, center=center_3s),
    label='3PWFS', color=colors['3s'], linewidth=2
)
plt.semilogy(
    circavg(psf_norm_4s / maxi, center=center_4s),
    label='4PWFS', color=colors['4s'], linewidth=2
)

# AO cutoff reference line
plt.axvline(
    papyrus.Nact / 2 * sampling_3s,
    color='k', linestyle=':', linewidth=1.5,
    label='AO cutoff'
)

# Aesthetics
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0, 50)
plt.ylim(1e-6, 1)

plt.xlabel('Radial distance [pix]', fontsize=16)
plt.ylabel('Normalized intensity', fontsize=16)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
plt.savefig(
    '/Volumes/SanDisk/PAPYRUS_run2/01.10.25/record/Radial_profile_transp.png',
    bbox_inches='tight', dpi=300, transparent=True
)
plt.show()

print("\n✅ Done — both PSFs processed, fitted, and compared.")
