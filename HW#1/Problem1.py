import numpy as np
import matplotlib.pyplot as plt
import os

# Create result directory if it doesn't exist
os.makedirs("./result", exist_ok=True)

harmonics = 10


# Define the square wave function over one period
def square_wave(x):
    return np.where((x % 1) < 0.5, 1, 0)


# Fourier series approximation of the square wave
def fourier_square(x, harmonics):
    # Start with the constant term a0 = 1/2
    f_approx = 0.5 * np.ones_like(x)
    for k in range(harmonics):
        n = 2 * k + 1  #  odd only
        f_approx += (2 / (np.pi * n)) * np.sin(2 * np.pi * n * x)
    return f_approx


# Create an x array over two periods for better visualization
x = np.linspace(0, 2, 1000)
y_square = square_wave(x)


# Plot the original square wave and its Fourier approximation
plt.figure(figsize=(10, 5))
plt.plot(
    x,
    y_square,
    label="Original Square Wave",
    color="black",
    linewidth=2,
    linestyle="--",
)
for harmonics in [1, 3, 5, 10, 20]:
    y_fourier = fourier_square(x, harmonics=harmonics)
    plt.plot(x, y_fourier, label=f"{harmonics} Harmonics")


plt.title("Square Wave and Its Fourier Series Approximation", fontsize=16)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("./result/square_wave_fourier.png", dpi=600, bbox_inches="tight")
plt.show()
