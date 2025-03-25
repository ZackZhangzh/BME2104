import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


# 加载和准备图像
def load_and_prepare_image(image_path, target_size=(512, 512)):
    """加载图像，转为灰度，并调整至512x512"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 转为灰度图
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # 调整大小到512x512
        img_resized = cv2.resize(img_gray, target_size)

        # 确保是8位图像
        if img_resized.dtype != np.uint8:
            img_resized = img_resized.astype(np.uint8)

        return img_resized
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return create_test_image()


def create_test_image():
    """创建一个带噪声的测试图像"""
    print("创建一个带噪声的测试图像...")
    img = np.zeros((512, 512), dtype=np.uint8)
    # 添加图案
    cv2.rectangle(img, (100, 100), (400, 400), 180, -1)
    cv2.circle(img, (250, 250), 150, 120, -1)

    # 添加高斯噪声
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy_img = cv2.add(img.astype(np.int16), noise)
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    return noisy_img


# 频域滤波实现
def frequency_filter(image, cutoff_radius):
    """在频域空间应用低通滤波器"""
    # 傅里叶变换
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)

    # 创建频域滤波器
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

    # 高斯低通滤波器
    mask = np.exp(-(distance**2) / (2 * cutoff_radius**2))

    # 应用滤波器
    filtered_fshift = f_shifted * mask

    # 逆变换回空间域
    f_ishift = np.fft.ifftshift(filtered_fshift)
    filtered_image = np.fft.ifft2(f_ishift)
    filtered_image = np.abs(filtered_image).astype(np.uint8)

    return filtered_image, mask


# 空间域滤波实现
def spatial_filter(image, kernel_size):
    """在图像空间应用平滑滤波器"""
    # 创建高斯核
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8  # 根据核大小计算合适的sigma
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T  # 2D高斯核

    # 应用滤波器
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image, kernel


# 计算评价指标
def calculate_metrics(original, filtered):
    """计算MSE, PSNR和SSIM"""
    mse = mean_squared_error(original, filtered)
    psnr = peak_signal_noise_ratio(original, filtered)
    ssim_value = structural_similarity(original, filtered)

    return mse, psnr, ssim_value


# 显示结果
def show_results(original, freq_filtered, spatial_filtered, freq_mask, spatial_kernel):
    plt.figure(figsize=(18, 10))

    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("原始噪声图像")
    plt.axis("off")

    # 频域滤波结果
    plt.subplot(2, 3, 2)
    plt.imshow(freq_filtered, cmap="gray")
    plt.title("频域滤波结果")
    plt.axis("off")

    # 空间滤波结果
    plt.subplot(2, 3, 3)
    plt.imshow(spatial_filtered, cmap="gray")
    plt.title("空间滤波结果")
    plt.axis("off")

    # 频域滤波器
    plt.subplot(2, 3, 5)
    plt.imshow(freq_mask, cmap="jet")
    plt.title("频域滤波器")
    plt.colorbar()

    # 空间滤波器
    plt.subplot(2, 3, 6)
    plt.imshow(spatial_kernel, cmap="jet")
    plt.title("空间滤波器")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("denoising_comparison.png", dpi=300)
    plt.show()


def main():
    # 加载图像
    try:
        # 尝试加载自定义图像文件
        image_path = "noisy_image.jpg"  # 更改为您的图像路径
        original_image = load_and_prepare_image(image_path)
    except:
        # 如果失败，使用生成的测试图像
        original_image = create_test_image()

    # 保存原始图像
    cv2.imwrite("original_noisy.png", original_image)

    # 1. 频域滤波 - 尝试不同截止频率
    cutoff_radii = [20, 40, 80]
    freq_results = []
    freq_masks = []

    for radius in cutoff_radii:
        filtered_img, mask = frequency_filter(original_image, radius)
        freq_results.append(filtered_img)
        freq_masks.append(mask)

        # 计算并显示指标
        mse, psnr, ssim_val = calculate_metrics(original_image, filtered_img)
        print(f"\n频域滤波 (截止半径={radius}):")
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim_val:.4f}")

        # 保存图像
        cv2.imwrite(f"freq_filtered_r{radius}.png", filtered_img)

    # 2. 空间滤波 - 尝试不同核大小
    kernel_sizes = [3, 5, 9]
    spatial_results = []
    spatial_kernels = []

    for size in kernel_sizes:
        filtered_img, kernel = spatial_filter(original_image, size)
        spatial_results.append(filtered_img)
        spatial_kernels.append(kernel)

        # 计算并显示指标
        mse, psnr, ssim_val = calculate_metrics(original_image, filtered_img)
        print(f"\n空间滤波 (核大小={size}x{size}):")
        print(f"  MSE: {mse:.2f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim_val:.4f}")

        # 保存图像
        cv2.imwrite(f"spatial_filtered_k{size}.png", filtered_img)

    # 3. 可视化结果对比 (使用中间参数)
    show_results(
        original_image,
        freq_results[1],  # 中等截止频率
        spatial_results[1],  # 中等核大小
        freq_masks[1],
        spatial_kernels[1],
    )


if __name__ == "__main__":
    main()
