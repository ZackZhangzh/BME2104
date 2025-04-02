import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftshift
import os
from scipy.interpolate import interp1d

# Create result directory if it doesn't exist
os.makedirs("./result", exist_ok=True)

# Read the image
img = plt.imread("pic/MTF-slice.tif")

# Get the center row of the image
line_profile = img[img.shape[0] // 2, :]

# Create x-axis in physical units (microns)
x = np.arange(len(line_profile)) * 7.6  # 7.6 microns per pixel

# print(f"Line profile length: {len(line_profile)}")
# print(x)

# Define Gaussian function for fitting
def gaussian(x, amplitude, mean, sigma, offset):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))+ offset


# Fit Gaussian to LIP
popt, _ = curve_fit(gaussian, x, line_profile)
fitted_gaussian = gaussian(x, *popt)

# Plot LIP and Gaussian fit
plt.figure(figsize=(10, 5))
plt.plot(x, line_profile, "b-", label="Line Intensity Profile")
plt.plot(x, fitted_gaussian, "r--", label="Gaussian Fit")
plt.title("Line Intensity Profile with Gaussian Fit")
plt.xlabel("Distance (μm)")
plt.ylabel("Intensity")

# Add Gaussian parameters to the plot
params_text = f"A: {popt[0]:.2f}\nμ: {popt[1]:.2f} um\nσ: {popt[2]:.2f} μm\nA0: {popt[3]:.2f}"
plt.annotate(params_text, xy=(0.02, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             verticalalignment='top')

plt.legend()
plt.grid(True)
plt.savefig("./result/lip_gaussian_fit.png", dpi=300, bbox_inches="tight")
plt.show()




# MTF

lsf = line_profile - popt[3]

# 使用 FFT 计算 FT，注意：FFT 计算的是离散傅里叶变换
ft = np.fft.fft(lsf)
ft = np.fft.fftshift(ft)  # 将直流分量移至中心
mtf = np.abs(ft)
mtf = mtf / mtf.max()  # 归一化

# 计算对应的空间频率（单位：cycles/mm）
pixel_size_mm = 7.6e-3  # 7.6 μm = 7.6e-3 mm
n = lsf.size #pixel number
freq = np.fft.fftfreq(n, d=pixel_size_mm)
freq = np.fft.fftshift(freq)
# 只取正频率部分进行拟合
positive = freq >= 0
freq_pos = freq[positive]
mtf_pos = mtf[positive]

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))

# initial_guess_mtf = [1, 0.5, 0]

popt_mtf, _ = curve_fit(gaussian, freq_pos, mtf_pos)

# Generate a finer frequency range for the Gaussian fit
f_fit = np.linspace(0, freq_pos.max(), 500)

# Compute the Gaussian fit for the MTF
mtf_fit = gaussian(f_fit, *popt_mtf)

# Calculate 10% MTF
# For a Gaussian function: finding where amplitude * exp(-((x - mean)^2)/(2*sigma^2)) = 0.1
# Solving for x gives: x = mean ± sigma * sqrt(-2*ln(0.1/amplitude))
mtf_10_percent = 0.1
if popt_mtf[0] > 0:  # Make sure amplitude is positive
    # Calculate the frequency at 10% MTF (taking only the positive solution)
    freq_at_10_mtf = popt_mtf[1] + popt_mtf[2] * np.sqrt(-2 * np.log(mtf_10_percent / popt_mtf[0]))
else:
    # Alternative method using interpolation if the Gaussian model doesn't fit well
    interp_func = interp1d(mtf_fit, f_fit, bounds_error=False, fill_value="extrapolate")
    freq_at_10_mtf = interp_func(0.1)





# Compute FFT of the fitted Gaussian (after subtracting offset)
gaussian_for_fft = fitted_gaussian - popt[3]
ft_gaussian = np.fft.fft(gaussian_for_fft)
ft_gaussian = np.fft.fftshift(ft_gaussian)
mtf_gaussian = np.abs(ft_gaussian)
mtf_gaussian = mtf_gaussian / mtf_gaussian.max()  # Normalize

# Extract the positive frequency part
mtf_gaussian_pos = mtf_gaussian[positive]

# Calculate 10% MTF for the Gaussian FFT
interp_func_gaussian = interp1d(freq_pos, mtf_gaussian_pos, bounds_error=False, fill_value="extrapolate")
# Find the frequency where MTF equals 0.1
freq_indices = np.where(mtf_gaussian_pos >= 0.1)[0]
if len(freq_indices) > 0:
    last_index = freq_indices[-1]
    if last_index < len(freq_pos) - 1:
        # Interpolate between points to find more accurate 10% MTF frequency
        x1, x2 = freq_pos[last_index], freq_pos[last_index + 1]
        y1, y2 = mtf_gaussian_pos[last_index], mtf_gaussian_pos[last_index + 1]
        freq_at_10_mtf_gaussian = x1 + (0.1 - y1) * (x2 - x1) / (y2 - y1)
    else:
        freq_at_10_mtf_gaussian = freq_pos[last_index]
else:
    # Fallback if no direct intersection is found
    freq_at_10_mtf_gaussian = interp_func_gaussian(0.1)

# Plot the MTF data and the Gaussian fit
plt.figure(figsize=(8, 4))
plt.plot(freq_pos, mtf_pos, 'bo', markersize=3, label='MTF Data')
plt.plot(f_fit, mtf_fit, 'r-', label='Gaussian Fit')
plt.plot(freq_pos, mtf_gaussian_pos, color='orange', linestyle='--', label='LIP (Gaussian Fit FFT)')

# Mark the 10% MTF point
plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.5)
plt.axvline(x=freq_at_10_mtf, color='g', linestyle='--', alpha=0.5)
plt.plot(freq_at_10_mtf, 0.1, 'g^', markersize=8, label=f'10% MTF')

# Add Gaussian parameters to the plot
params_text_mtf = f"A: {popt_mtf[0]:.2f}\nμ: {popt_mtf[1]:.2f} cycles/mm\nσ: {popt_mtf[2]:.2f} cycles/mm\n10% MTF: {freq_at_10_mtf:.2f} cycles/mm"
plt.annotate(params_text_mtf, xy=(0.02, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
             verticalalignment='top')

# Label the axes and add a title
plt.xlabel('Spatial Frequency (cycles/mm)')
plt.ylabel('MTF')
plt.title('MTF Gaussian Fit with 10% MTF')

# Add a legend
plt.legend()

# Save the plot to the result directory
plt.savefig("./result/mtf_gaussian_fit.png", dpi=300, bbox_inches="tight")
plt.show()




