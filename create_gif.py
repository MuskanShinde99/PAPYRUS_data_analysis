from astropy.io import fits
import imageio
import numpy as np

def fits_cube_to_gif_fast(fits_path, gif_path, cmap='gray', duration=0.1):
    with fits.open(fits_path) as hdul:
        data_cube = hdul[0].data

    images = []
    for i in range(data_cube.shape[0]):
        frame = data_cube[i]

        # Optional: rescale each frame linearly for display
        frame_min, frame_max = np.nanmin(frame), np.nanmax(frame)
        if frame_max > frame_min:
            norm_frame = (frame - frame_min) / (frame_max - frame_min)
        else:
            norm_frame = np.zeros_like(frame)

        # Convert to 8-bit image for GIF
        frame_8bit = (norm_frame * 255).astype(np.uint8)
        images.append(frame_8bit)

    # Save to GIF
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved to: {gif_path}")

# Example usage
fits_file = '/Volumes/UNIGE/RISTRETTO/PAPYTUS_tests_results/AObench/src/record/2025-08-08_18-16-28/cblue.fits'
output_gif = '/Volumes/UNIGE/RISTRETTO/PAPYTUS_tests_results/seeing_3arcsec_4-sided_PWFS_closedloop.gif'
fits_cube_to_gif_fast(fits_file, output_gif)

#seeing_1arcsec_3-sided_PWFS_closedloop