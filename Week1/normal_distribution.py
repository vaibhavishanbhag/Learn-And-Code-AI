import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data for Normal Distribution
mean = 0       # Mean (center)
std_dev = 1    # Standard Deviation
x = np.linspace(-4, 4, 1000)  # Range of values
y = norm.pdf(x, mean, std_dev)  # Probability Density Function (PDF)

# Plot the Normal Distribution Curve
plt.plot(x, y, color='blue', label='Normal Distribution')
plt.fill_between(x, y, alpha=0.3, color='blue')  # Fill area under curve

# Marking Mean and Standard Deviations
plt.axvline(mean, color='red', linestyle='dashed', label="Mean (μ)")
plt.axvline(mean - std_dev, color='green', linestyle='dashed', label="μ - σ")
plt.axvline(mean + std_dev, color='green', linestyle='dashed', label="μ + σ")

# Labels & Title
plt.title("Normal Distribution (Gaussian Curve)")
plt.xlabel("Values")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
