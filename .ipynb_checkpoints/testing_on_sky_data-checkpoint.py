#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:27:22 2025

@author: muskanshinde
"""

import os
import numpy as np
from astropy.io import fits
import imageio.v2 as iio
from matplotlib import pyplot as plt

# ---- your folder ----
pathname = '/Users/muskanshinde/Downloads/wetransfer_3pwfs-cl-2025-09-11t06_41_33-cube-fits_2025-09-15_1319'

# ---- input FITS (raw cubes) ----
fn_ol = '3PWFS-move-OL-2025-09-11T06_44_22-Cube.fits'
fn_cl = '3PWFS-CL-2025-09-11T06_41_33-Cube.fits'

# ---- load & dark-subtract using last frame (no normalization) ----
OL_cube = fits.getdata(os.path.join(pathname, fn_ol)).astype(float)
CL_cube = fits.getdata(os.path.join(pathname, fn_cl)).astype(float)
OL_cube_dark_subs = OL_cube - OL_cube[-1][None, ...]
CL_cube_dark_subs = CL_cube - CL_cube[-1][None, ...]


def robust_vrange(cube, low=0.5, high=99.5, crop=0.05):
    """
    One global vmin/vmax per cube using robust percentiles.
    'crop' ignores a border fraction to avoid hot edges.
    """
    T, H, W = cube.shape
    y0 = int(H * crop); x0 = int(W * crop)
    y1 = max(y0 + 1, H - y0)  # ensure at least 1 px
    x1 = max(x0 + 1, W - x0)
    sub = cube[:, y0:y1, x0:x1].ravel()
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        return 0.0, 1.0
    vmin = np.percentile(sub, low)
    vmax = np.percentile(sub, high)
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0
    return float(vmin), float(vmax)

def to_uint8_gamma(cube, vmin=None, vmax=None, gamma=0.5):
    """
    Linear map to [0,1] with ONE global vmin/vmax, apply gamma (sqrt if 0.5),
    then map to uint8. No per-frame changes.
    """
    if vmin is None or vmax is None:
        vmin, vmax = robust_vrange(cube)
    lin = (cube - vmin) / (vmax - vmin)
    lin = np.clip(lin, 0.0, 1.0)
    if gamma != 1.0:
        lin = np.power(lin, gamma)
    return (lin * 255.0 + 0.5).astype(np.uint8), vmin, vmax

def save_gif(cube, outpath, fps=15, vmin=None, vmax=None, gamma=0.5):
    """
    Save GIF with ALL frames (no decimation). Uses global vmin/vmax and gamma.
    """
    cube_u8, vmin_used, vmax_used = to_uint8_gamma(cube, vmin=vmin, vmax=vmax, gamma=gamma)
    frames = [cube_u8[k] for k in range(cube_u8.shape[0])]
    iio.mimsave(outpath, frames, duration=1.0/float(fps), loop=0)
    print(f"Saved {outpath}  ({len(frames)} frames @ {fps} fps)  "
          f"[vmin={vmin_used:.3g}, vmax={vmax_used:.3g}, gamma={gamma}]")

# -------- compute robust scales (tweak 'high' to brighten faint detail) --------
# OL often needs stronger clipping of outliers:
vmin_ol, vmax_ol = robust_vrange(OL_cube_dark_subs, low=0.5, high=99.0, crop=0.05)
vmin_cl, vmax_cl = robust_vrange(CL_cube_dark_subs, low=0.5, high=99.5, crop=0.05)

# -------- write GIFs (sqrt = gamma 0.5) --------
ol_gif = os.path.join(pathname, 'OL_dark_subtracted_sqrt.gif')
cl_gif = os.path.join(pathname, 'CL_dark_subtracted_sqrt.gif')

save_gif(OL_cube_dark_subs, ol_gif, fps=15, vmin=vmin_ol, vmax=vmax_ol, gamma=0.5)
save_gif(CL_cube_dark_subs, cl_gif, fps=15, vmin=vmin_cl, vmax=vmax_cl, gamma=0.5)

print("Done.")






#%%

# Plot
fig = plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
im1 = ax1.imshow(np.log10(np.median(OL_cube_dark_subs, axis=0)), cmap='gray')
ax1.set_title('OL cube average img')
fig.colorbar(im1, ax=ax1)

ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(np.log10(np.median(CL_cube_dark_subs, axis=0)), cmap='gray')
ax2.set_title('CL cube average img')
fig.colorbar(im2, ax=ax2)

plt.tight_layout()

# Save (no timestamp)
fig.savefig(os.path.join(pathname, "OL_CL_average_image.png"), dpi=300, bbox_inches='tight')
plt.show()

#%%
plt.figure()
plt.imshow((np.std(CL_cube_dark_subs,axis=0))[5:15, 135:145], cmap='gray')
plt.title('CL cube')
plt.colorbar()

plt.figure()
plt.imshow((np.std(CL_cube_dark_subs,axis=0))[5:35, 125:155] , cmap='gray')
plt.title('CL cube')
plt.colorbar()

#[:, 5:35, 125:155]  

#%%
CL_cube_dark_subs = CL_cube_dark_subs[:,5:35, 125:155]  
OL_cube_dark_subs =  OL_cube_dark_subs[:, 5:35, 125:155]  

#%%

OL_FFT2D =  [np.fft.fftshift(np.fft.fft2(OL_cube_dark_subs[i, :, :])) for i in range(0, 1000)]
OL_mag = np.abs(OL_FFT2D) 


CL_FFT2D =  [np.fft.fftshift(np.fft.fft2(CL_cube_dark_subs[i, :, :])) for i in range(0, 1000)]
CL_mag = np.abs(CL_FFT2D) 

plt.figure() 
plt.subplot(121)
plt.imshow(np.log10(OL_mag)[10])
plt.colorbar()
plt.title("2D FFT OL cube")

plt.subplot(122)
plt.imshow(np.log10(CL_mag)[10])
plt.colorbar()
plt.title("2D FFT CL cube")
plt.show()



plt.figure()
plt.subplot(121)
plt.imshow(np.mean(np.log10(OL_mag),axis=0))
plt.colorbar()
plt.title("Mean 2D FFT OL cube")

plt.subplot(122)
plt.imshow(np.mean(np.log10(CL_mag),axis=0))
plt.colorbar()
plt.title("Mean 2D FFT CL cube")
plt.show()


# plt.figure() 
# plt.imshow(OL_FFT2D)
# plt.show()


