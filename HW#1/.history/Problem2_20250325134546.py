import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftshift
import os

# Create result directory if it doesn't exist
os.makedirs("./result", exist_ok=True)

# Read the image
img = plt.imread("MTF-slice.tif")

# Get the center row of the image
center_row = img[img.shape[0] // 2, :]

# Create x-axis in physical units (microns)
x = np.arange(len(center_row)) * 7.6  # 7.6 microns per pixel


# Define Gaussian function for fitting
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


# Fit Gaussian to LIP
popt, _ = curve_fit(gaussian, x, center_row)
fitted_gaussian = gaussian(x, *popt)

# Plot LIP and Gaussian fit
plt.figure(figsize=(10, 5))
plt.plot(x, center_row, "b-", label="Line Intensity Profile")
plt.plot(x, fitted_gaussian, "r--", label="Gaussian Fit")
plt.title("Line Intensity Profile with Gaussian Fit")
plt.xlabel("Distance (μm)")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)
plt.savefig("./result/lip_gaussian_fit.png", dpi=300, bbox_inches="tight")
plt.show()

# Calculate MTF using FFT of LSF
# First, normalize the LSF
lsf = center_row / np.max(center_row)

# Calculate FFT
fft_result = fft(lsf)
fft_magnitude = np.abs(fft_result)
fft_magnitude = fftshift(fft_magnitude)  # Shift zero frequency to center

# Create frequency axis
freq = np.fft.fftshift(np.fft.fftfreq(len(lsf), d=7.6))  # d is pixel size in microns
freq = freq[: len(freq) // 2]  # Take only positive frequencies
mtf = fft_magnitude[len(fft_magnitude) // 2 :]  # Take only positive frequencies


# Fit MTF with Gaussian
def gaussian_mtf(f, amplitude, sigma):
    return amplitude * np.exp(-((2 * np.pi * f * sigma) ** 2) / 2)


# Fit Gaussian to MTF
popt_mtf, _ = curve_fit(gaussian_mtf, freq, mtf)
fitted_mtf = gaussian_mtf(freq, *popt_mtf)

# Plot MTF and Gaussian fit
plt.figure(figsize=(10, 5))
plt.plot(freq, mtf, "b-", label="MTF")
plt.plot(freq, fitted_mtf, "r--", label="Gaussian Fit")
plt.title("Modulation Transfer Function with Gaussian Fit")
plt.xlabel("Spatial Frequency (cycles/mm)")
plt.ylabel("MTF")
plt.legend()
plt.grid(True)
plt.savefig("./result/mtf_gaussian_fit.png", dpi=300, bbox_inches="tight")
plt.show()

# Find spatial resolution at 10% MTF
mtf_10 = 0.1
freq_10 = freq[np.where(mtf >= mtf_10)[0][-1]]
resolution_10 = 1 / freq_10  # in mm

print(f"Spatial resolution at 10% MTF: {resolution_10:.2f} mm")

# Compare Gaussian parameters
print("\nGaussian parameters:")
print(f"LSF Gaussian sigma: {popt[2]:.2f} μm")
print(f"MTF Gaussian sigma: {popt_mtf[1]:.2f} cycles/mm")

# Theoretical relationship check
# If LSF is Gaussian with sigma_s, then MTF should be Gaussian with sigma_f = 1/(2π*sigma_s)
theoretical_sigma = 1 / (2 * np.pi * popt[2])
print(f"\nTheoretical MTF sigma: {theoretical_sigma:.2f} cycles/mm")
print(f"Measured MTF sigma: {popt_mtf[1]:.2f} cycles/mm")
