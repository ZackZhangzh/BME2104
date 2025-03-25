import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity as ssim
import os
from PIL import Image

# Create result directory if it doesn't exist
os.makedirs("./result", exist_ok=True)


def load_and_preprocess_image(image_path):
    """Load and preprocess image to 512x512 grayscale."""
    # Read image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    # Resize to 512x512
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    return np.array(img)


def create_gaussian_lowpass_filter(shape, cutoff_freq):
    """Create a Gaussian low-pass filter."""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[-center_row : rows - center_row, -center_col : cols - center_col]
    mask = x * x + y * y <= cutoff_freq * cutoff_freq
    return mask.astype(float)


def create_gaussian_kernel(size=5, sigma=1.0):
    """Create a 2D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)
        ),
        (size, size),
    )
    return kernel / kernel.sum()


def calculate_metrics(original, denoised):
    """Calculate MSE, PSNR, and SSIM."""
    mse = np.mean((original - denoised) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    ssim_value = ssim(original, denoised, data_range=255)
    return mse, psnr, ssim_value


def frequency_domain_denoising(image, cutoff_freqs=[50, 100, 150]):
    """Perform frequency domain denoising with different cutoff frequencies."""
    # Compute 2D FFT
    fft_img = fft2(image)
    fft_shifted = fftshift(fft_img)

    denoised_images = []
    for cutoff in cutoff_freqs:
        # Create and apply low-pass filter
        filter_mask = create_gaussian_lowpass_filter(image.shape, cutoff)
        filtered_fft = fft_shifted * filter_mask

        # Inverse FFT
        denoised = np.abs(ifft2(ifftshift(filtered_fft)))
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        denoised_images.append(denoised)

    return denoised_images


def spatial_domain_denoising(image, kernel_sizes=[3, 5, 7]):
    """Perform spatial domain denoising with different kernel sizes."""
    denoised_images = []
    for size in kernel_sizes:
        kernel = create_gaussian_kernel(size=size, sigma=size / 3)
        denoised = convolve(image, kernel, mode="reflect")
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        denoised_images.append(denoised)

    return denoised_images


def plot_results(original, denoised_freq, denoised_spatial, cutoff_freqs, kernel_sizes):
    """Plot original and denoised images."""
    # Plot frequency domain results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    for i, (img, cutoff) in enumerate(zip(denoised_freq, cutoff_freqs)):
        plt.subplot(1, 3, i + 2)
        plt.imshow(img, cmap="gray")
        plt.title(f"Frequency Domain\nCutoff={cutoff}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("./result/frequency_domain_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot spatial domain results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    for i, (img, size) in enumerate(zip(denoised_spatial, kernel_sizes)):
        plt.subplot(1, 3, i + 2)
        plt.imshow(img, cmap="gray")
        plt.title(f"Spatial Domain\nKernel Size={size}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("./result/spatial_domain_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    # Load and preprocess image
    # Replace 'your_image.jpg' with your image path
    image = load_and_preprocess_image("your_image.jpg")

    # Parameters for denoising
    cutoff_freqs = [50, 100, 150]
    kernel_sizes = [3, 5, 7]

    # Perform denoising
    denoised_freq = frequency_domain_denoising(image, cutoff_freqs)
    denoised_spatial = spatial_domain_denoising(image, kernel_sizes)

    # Plot results
    plot_results(image, denoised_freq, denoised_spatial, cutoff_freqs, kernel_sizes)

    # Calculate and print metrics
    print("\nFrequency Domain Denoising Metrics:")
    for i, (denoised, cutoff) in enumerate(zip(denoised_freq, cutoff_freqs)):
        mse, psnr, ssim_value = calculate_metrics(image, denoised)
        print(f"\nCutoff Frequency = {cutoff}:")
        print(f"MSE: {mse:.2f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")

    print("\nSpatial Domain Denoising Metrics:")
    for i, (denoised, size) in enumerate(zip(denoised_spatial, kernel_sizes)):
        mse, psnr, ssim_value = calculate_metrics(image, denoised)
        print(f"\nKernel Size = {size}:")
        print(f"MSE: {mse:.2f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")


if __name__ == "__main__":
    main()
