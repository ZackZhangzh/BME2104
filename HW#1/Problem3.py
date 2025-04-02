import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage 
from PIL import Image
from scipy import fftpack
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Create result directory if it doesn't exist
os.makedirs("./result", exist_ok=True)
image_path='./pic/problem3.jpg'

def load_and_preprocess_image(image_path):
    """Load and preprocess image to 512x512 grayscale."""
    # Read image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    # Resize to 512x512
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    return np.array(img)

def frequency_domain_filter(image, cutoff_frequency):
    """Apply low-pass filter in frequency domain with given cutoff frequency."""
    # 执行傅里叶变换
    f_transform = fftpack.fft2(image)
    f_transform_shifted = fftpack.fftshift(f_transform)
    
    # 创建低通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 1
    
    # 应用滤波器
    f_transform_filtered = f_transform_shifted * mask
    f_transform_filtered_back = fftpack.ifftshift(f_transform_filtered)
    
    # 执行逆傅里叶变换
    img_filtered = fftpack.ifft2(f_transform_filtered_back)
    img_filtered = np.abs(img_filtered)
    
    return img_filtered

def spatial_domain_filter(image, kernel_size):
    """Apply spatial domain filtering with a mean filter of given kernel size."""
    # 创建均值滤波器
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # 执行卷积
    filtered_image = scipy.ndimage.convolve(image, kernel, mode='reflect')
    
    return filtered_image

def calculate_metrics(original, filtered):
    """Calculate MSE, PSNR and SSIM between original and filtered images."""
    # 确保图像类型一致且计算数据范围
    original = original.astype(np.float64)
    filtered = filtered.astype(np.float64)
    
    # 计算数据范围
    data_range = 255.0  # 灰度图像的通常范围
    
    # 计算MSE
    mse = np.mean((original - filtered) ** 2)
    
    # 计算PSNR和SSIM，指定data_range参数
    psnr_value = psnr(original, filtered, data_range=data_range)
    ssim_value = ssim(original, filtered, data_range=data_range)
    
    return mse, psnr_value, ssim_value

def plot_results(images, titles, filename):
    """Plot multiple images with titles and save figure."""
    fig, axes = plt.subplots(1, len(images), figsize=(16, 6))
    
    # Reduce space between subplots and borders
    plt.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.9, bottom=0.02)
    
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(title, fontsize=26)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./result/{filename}.png')
    plt.close()

def main():
    # 加载并预处理图像
    image = load_and_preprocess_image(image_path)
    
    # 显示原始图像
    plt.figure(frameon=False)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('./result/original_image.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 1. 频域滤波 - 尝试不同的截止频率
    cutoff_frequencies = [30, 50, 80]
    freq_filtered_images = []
    freq_results = []
    
    for cutoff in cutoff_frequencies:
        filtered = frequency_domain_filter(image, cutoff)
        freq_filtered_images.append(filtered)
        
        mse, psnr_value, ssim_value = calculate_metrics(image, filtered)
        freq_results.append((cutoff, mse, psnr_value, ssim_value))
    
    # 绘制频域滤波结果
    titles = [f'Low-pass (cutoff={cutoff})' for cutoff in cutoff_frequencies]
    plot_results(freq_filtered_images, titles, 'frequency_domain_results')
    
    # 2. 空域滤波 - 尝试不同大小的滤波器核
    kernel_sizes = [3, 5, 7]
    spatial_filtered_images = []
    spatial_results = []
    
    for k_size in kernel_sizes:
        filtered = spatial_domain_filter(image, k_size)
        spatial_filtered_images.append(filtered)
        
        mse, psnr_value, ssim_value = calculate_metrics(image, filtered)
        spatial_results.append((k_size, mse, psnr_value, ssim_value))
    
    # 绘制空域滤波结果
    titles = [f'Mean Filter (kernel={k_size}x{k_size})' for k_size in kernel_sizes]
    plot_results(spatial_filtered_images, titles, 'spatial_domain_results')
    
    # 打印评估指标
    print("频域滤波结果:")
    print("截止频率 | MSE | PSNR | SSIM")
    print("-" * 50)
    for result in freq_results:
        print(f"{result[0]:^8} | {result[1]:.2f} | {result[2]:.2f} | {result[3]:.4f}")
    
    print("\n空域滤波结果:")
    print("核大小 | MSE | PSNR | SSIM")
    print("-" * 50)
    for result in spatial_results:
        print(f"{result[0]:^6} | {result[1]:.2f} | {result[2]:.2f} | {result[3]:.4f}")
    
    # 比较最佳结果
    best_freq = freq_filtered_images[np.argmax([r[3] for r in freq_results])]
    best_spatial = spatial_filtered_images[np.argmax([r[3] for r in spatial_results])]
    
    plot_results([image, best_freq, best_spatial], 
                ['Original', 'Best Frequency Domain', 'Best Spatial Domain'], 
                'comparison')

if __name__ == "__main__":
    main()