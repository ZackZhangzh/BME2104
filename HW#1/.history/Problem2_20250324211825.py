import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import fft
import cv2
from scipy.ndimage import center_of_mass

# 定义高斯函数模型用于拟合
def gaussian(x, A, mu, sigma, c):
    """高斯函数
    
    Parameters:
    x: x坐标
    A: 幅度
    mu: 均值(中心)
    sigma: 标准差
    c: 常数(基线)
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

# MTF的高斯模型
def gaussian_mtf(f, A, sigma, c):
    """高斯MTF函数
    
    Parameters:
    f: 空间频率
    A: 幅度
    sigma: 与LIP高斯函数的sigma相关的参数
    c: 常数(基线)
    """
    return A * np.exp(-2 * (np.pi * sigma * f)**2) + c

# 读取图像文件
try:
    image = cv2.imread("MTF-slice.tif", cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = cv2.imread("MTF-slice.jpg", cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = cv2.imread("MTF-slice.png", cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError("无法找到图像文件，请检查格式和路径。")
except Exception as e:
    print(f"读取图像时出错: {e}")
    exit(1)

# 像素大小(μm)
pixel_size = 7.6

# 显示原始图像
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.colorbar()
plt.savefig('original_image.png')
plt.show()

# 查找中心区域的钨丝
# 使用阈值处理突出钨丝
_, thresh = cv2.threshold(image, 0.8 * np.max(image), 255, cv2.THRESH_BINARY)
# 找到钨丝的质心位置
cy, cx = center_of_mass(thresh)
cy, cx = int(cy), int(cx)

print(f"钨丝中心位置: x={cx}, y={cy}")

# 显示图像和标记出的钨丝中心
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.plot(cx, cy, 'r+', markersize=15)
plt.title('标记的钨丝中心')
plt.colorbar()
plt.savefig('wire_center.png')
plt.show()

# 提取水平线强度曲线(LIP)
horizontal_profile = image[cy, :]

# 确定拟合范围（围绕钨丝中心）
window_size = 100  # 根据实际图像调整
h_range = slice(max(0, cx - window_size//2), min(cx + window_size//2, image.shape[1]))

x_h = np.arange(h_range.start, h_range.stop)
y_h = horizontal_profile[h_range]

# 用高斯函数拟合水平线强度曲线
try:
    h_params, h_covariance = curve_fit(gaussian, x_h, y_h, 
                                      p0=[np.max(y_h)-np.min(y_h), cx, 5, np.min(y_h)])
    
    # 提取参数
    h_A, h_mu, h_sigma, h_c = h_params
    
    # 生成拟合曲线
    h_fit = gaussian(x_h, *h_params)
    
    # 绘制水平线强度曲线及其拟合
    plt.figure(figsize=(10, 6))
    plt.plot((x_h-cx)*pixel_size, y_h, 'bo', label='测量的LIP')
    plt.plot((x_h-cx)*pixel_size, h_fit, 'r-', label='高斯拟合')
    plt.xlabel('位置 (μm)')
    plt.ylabel('强度')
    plt.title('钨丝的线强度曲线(LIP)')
    plt.legend()
    plt.grid(True)
    plt.savefig('line_intensity_profile.png')
    plt.show()
    
    print("\nLIP高斯拟合参数:")
    print(f"A = {h_A:.2f}, mu = {h_mu:.2f} 像素, sigma = {h_sigma:.2f} 像素 ({h_sigma*pixel_size:.2f} μm), c = {h_c:.2f}")
    
except RuntimeError as e:
    print(f"LIP曲线拟合出错: {e}")
    exit(1)

# 计算MTF (假设LIP是LSF)
# 准备LSF数据，减去基线并归一化
lsf = h_fit - h_c
lsf = lsf / np.sum(lsf)

# 计算MTF (LSF的傅里叶变换的幅度)
mtf_raw = np.abs(fft.fftshift(fft.fft(lsf)))
# 归一化MTF
mtf = mtf_raw / mtf_raw.max()

# 计算空间频率轴(lp/mm)
freqs = fft.fftshift(fft.fftfreq(len(x_h), d=pixel_size*1e-3))  # 从μm转换为mm

# 仅保留正频率部分
positive_freq_idx = freqs >= 0
freqs = freqs[positive_freq_idx]
mtf = mtf[positive_freq_idx]

# 拟合MTF数据
try:
    mtf_params, mtf_covariance = curve_fit(
        gaussian_mtf, freqs, mtf, 
        p0=[1.0, h_sigma*pixel_size*1e-3, 0], 
        bounds=([0, 0, -0.1], [1.1, np.inf, 0.1])
    )
    
    A_mtf, sigma_mtf, c_mtf = mtf_params
    
    # 生成拟合曲线
    mtf_fit = gaussian_mtf(freqs, *mtf_params)
    
    # 计算理论MTF (基于LSF的高斯拟合)
    lsf_sigma_mm = h_sigma * pixel_size * 1e-3  # 转换为mm
    theoretical_mtf = gaussian_mtf(freqs, 1.0, lsf_sigma_mm, 0)
    
    # 绘制MTF
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mtf, 'b-', label='计算的MTF')
    plt.plot(freqs, mtf_fit, 'r--', label='高斯拟合')
    plt.plot(freqs, theoretical_mtf, 'g-.', label='从LSF理论推导的MTF')
    
    # 添加10% MTF线
    plt.axhline(y=0.1, color='k', linestyle=':')
    
    # 找到10% MTF处的空间频率
    # 使用插值获得更精确的值
    from scipy.interpolate import interp1d
    mtf_interp = interp1d(freqs, mtf, kind='cubic')
    
    # 创建更密集的频率点
    dense_freqs = np.linspace(freqs.min(), freqs.max(), 1000)
    dense_mtf = mtf_interp(dense_freqs)
    
    # 找到最接近0.1的MTF值
    mtf_10_idx = np.argmin(np.abs(dense_mtf - 0.1))
    f_10 = dense_freqs[mtf_10_idx]
    resolution_10 = 1 / f_10  # mm
    
    plt.axvline(x=f_10, color='k', linestyle=':')
    plt.text(f_10, 0.5, f'{f_10:.2f} lp/mm\n({resolution_10*1000:.2f} μm)', fontsize=10)
    
    plt.xlabel('空间频率 (lp/mm)')
    plt.ylabel('MTF')
    plt.title('调制传递函数 (MTF)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max(freqs))
    plt.ylim(0, 1.1)
    plt.savefig('mtf_plot.png')
    plt.show()
    
    # 比较LSF高斯拟合的sigma和MTF高斯拟合的sigma
    expected_mtf_sigma = lsf_sigma_mm
    
    print("\nMTF高斯拟合参数:")
    print(f"A = {A_mtf:.4f}, sigma = {sigma_mtf:.4f} mm, c = {c_mtf:.4f}")
    
    print("\n比较高斯参数:")
    print(f"LSF sigma: {h_sigma:.2f} 像素 或 {lsf_sigma_mm:.4f} mm")
    print(f"MTF拟合的sigma: {sigma_mtf:.4f} mm")
    print(f"比值 (MTF/LSF): {sigma_mtf/lsf_sigma_mm:.4f}")
    
    print(f"\n10% MTF处的空间分辨率: {f_10:.2f} lp/mm 或 {resolution_10*1000:.2f} μm")
    
except Exception as e:
    print(f"MTF拟合出错: {e}")