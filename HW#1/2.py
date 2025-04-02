import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.image import imread

# ---------------------------
# (1) LIP 提取及高斯拟合（LSF）
# ---------------------------
def gaussian(x, A, mu, sigma, offset):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2)) + offset

# 载入图像
image = imread("pic/MTF-slice.tif")
if image.ndim == 3:
    image = image.mean(axis=2)

# 提取图像中心行（LSF）
row_center = image.shape[0] // 2
line_profile = image[row_center, :]

# x 轴为像素索引
x = np.arange(line_profile.size)* 7.6 

# 初始猜测：A, mu, sigma, offset
initial_guess = [line_profile.max() - line_profile.min(), np.argmax(line_profile), 10, line_profile.min()]

popt, _ = curve_fit(gaussian, x, line_profile, p0=initial_guess)

# 计算拟合曲线
x_fit = np.linspace(0, x[-1], 500)
y_fit = gaussian(x_fit, *popt)

plt.figure(figsize=(8, 4))
plt.plot(x, line_profile, 'bo', markersize=3, label='原始 LSF')
plt.plot(x_fit, y_fit, 'r-', label='高斯拟合')
plt.xlabel('像素')
plt.ylabel('强度')
plt.title('图像中心 LSF 及高斯拟合')
plt.legend()
plt.show()

print("LSF 高斯拟合参数：")
print(f"A = {popt[0]:.3f}, μ = {popt[1]:.3f}, σ = {popt[2]:.3f} pixel, offset = {popt[3]:.3f}")

# ---------------------------
# (2) 计算 MTF
# ---------------------------
# 为了得到 LSF 的 FT，先去除背景（offset）
lsf = line_profile - popt[3]

# 使用 FFT 计算 FT，注意：FFT 计算的是离散傅里叶变换
ft = np.fft.fft(lsf)
ft = np.fft.fftshift(ft)  # 将直流分量移至中心
mtf = np.abs(ft)
mtf = mtf / mtf.max()  # 归一化

# 计算对应的空间频率（单位：cycles/mm）
pixel_size_mm = 7.6e-3  # 7.6 μm = 7.6e-3 mm
n = lsf.size
freq = np.fft.fftfreq(n, d=pixel_size_mm)
freq = np.fft.fftshift(freq)
# 只取正频率部分进行拟合
positive = freq >= 0
freq_pos = freq[positive]
mtf_pos = mtf[positive]

plt.figure(figsize=(8, 4))
plt.plot(freq_pos, mtf_pos, 'bo', markersize=3, label='MTF (FFT 得到)')
plt.xlabel('空间频率 (cycles/mm)')
plt.ylabel('MTF')
plt.title('MTF 曲线')
plt.legend()
plt.show()

# ---------------------------
# (2) 对 MTF 进行高斯拟合
# ---------------------------
# 定义 MTF 高斯模型（理论上中心在 0）
def gaussian_mtf(f, A, sigma_f, offset):
    return A * np.exp(- (f)**2 / (2 * sigma_f**2)) + offset

# 初始猜测：A ~1, sigma_f ~? 这里可以取正频率中半高宽处作为估计，offset ~0
initial_guess_mtf = [1, 0.5, 0]

popt_mtf, _ = curve_fit(gaussian_mtf, freq_pos, mtf_pos, p0=initial_guess_mtf)

f_fit = np.linspace(0, freq_pos.max(), 500)
mtf_fit = gaussian_mtf(f_fit, *popt_mtf)

plt.figure(figsize=(8, 4))
plt.plot(freq_pos, mtf_pos, 'bo', markersize=3, label='MTF 数据')
plt.plot(f_fit, mtf_fit, 'r-', label='高斯拟合')
plt.xlabel('空间频率 (cycles/mm)')
plt.ylabel('MTF')
plt.title('MTF 高斯拟合')
plt.legend()
plt.show()

print("MTF 高斯拟合参数：")
print(f"A = {popt_mtf[0]:.3f}, σ_f = {popt_mtf[1]:.3f} cycles/mm, offset = {popt_mtf[2]:.3f}")

# ---------------------------
# (2) 理论讨论：FT 高斯与 MTF 拟合的关系
# ---------------------------
# 理论上，若 LSF = exp(- (x-μ)^2/(2σ^2)),
# 则其 FT (归一化后) ∝ exp(-2π²σ²f²).
# 所以理论上 σ_f,theory = 1/(2πσ_eff), 其中 σ_eff 为 LSF 去除采样因素后的标准差
# （注意此处 LSF 的 sigma 单位为 pixel，换算成 mm 需乘以 pixel_size_mm）
sigma_mm = popt[2] * pixel_size_mm
sigma_f_theory = 1 / (2 * np.pi * sigma_mm)
print(f"基于 LSF 拟合参数计算理论 σ_f = {sigma_f_theory:.3f} cycles/mm")

# ---------------------------
# (2) 计算 10% MTF 时的空间频率及分辨率
# ---------------------------
# 利用拟合的 MTF 高斯模型：设 MTF(f_10) = 0.1,
# 对于高斯模型: 0.1 = A * exp(- f_10^2/(2σ_f^2)) + offset. 若 A≈1, offset≈0，则
# f_10 = σ_f * sqrt(2 * ln(1/A/0.1)) = σ_f * sqrt(2 * ln(10)).
f_10 = popt_mtf[1] * np.sqrt(2 * np.log(10))
print(f"10% MTF 对应的空间频率 f_10 = {f_10:.3f} cycles/mm")

# 分辨率通常定义为 1/f_10（单位 mm），也可以转换为 μm
resolution_mm = 1 / f_10
resolution_um = resolution_mm * 1e3
print(f"10% MTF 定义的空间分辨率 = {resolution_mm:.3f} mm 或 {resolution_um:.1f} μm")
