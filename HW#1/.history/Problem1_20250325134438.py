import numpy as np
import matplotlib.pyplot as plt
import os

# Create result directory if it doesn't exist
os.makedirs("./result", exist_ok=True)


# Define the square wave function over one period
def square_wave(x):
    return np.where((x % 1) < 0.5, 1, 0)


# Fourier series approximation of the square wave
def fourier_square(x, harmonics=10):
    # Start with the constant term a0 = 1/2
    f_approx = 0.5 * np.ones_like(x)
    for k in range(harmonics):
        n = 2 * k + 1  # only odd harmonics
        f_approx += (2 / (np.pi * n)) * np.sin(2 * np.pi * n * x)
    return f_approx


# Create an x array over two periods for better visualization
x = np.linspace(0, 2, 1000)
y_square = square_wave(x)
y_fourier = fourier_square(x, harmonics=10)  # try with 10 odd harmonics

# Plot the original square wave and its Fourier approximation
plt.figure(figsize=(10, 5))
plt.plot(x, y_square, label="Original Square Wave", color="black", linewidth=2)
plt.plot(
    x, y_fourier, label="Fourier Series Approximation", linestyle="--", color="red"
)
plt.title("Square Wave and Its Fourier Series Approximation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("./result/square_wave_fourier.png", dpi=3600, bbox_inches="tight")
plt.show()
