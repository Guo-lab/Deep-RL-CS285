#!/usr/bin/env python3
"""
Plot variance comparison between original and return-to-go policy gradients
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define theta values avoiding 0 and 1 to prevent division by zero
theta = np.linspace(0.01, 0.99, 500)

# Variance of the original policy gradient
# variance_original = 1 / (theta * (1 - theta) ** 2)
variance_original = (1 + 8 * theta + 4 * theta**2) / (theta * (1 - theta) ** 4)

# Variance of the return-to-go policy gradient
variance_rtg = (1 + 3 * theta + theta**2) / (theta * (1 - theta) ** 4)

# Plot both variances
plt.figure(figsize=(10, 6))
plt.plot(
    theta,
    variance_original,
    label=r"Original: $\frac{1 + 8\theta + 4\theta^2}{\theta(1-\theta)^4}$",
    linewidth=2.5,
    color="blue",
)
plt.plot(
    theta,
    variance_rtg,
    label=r"Return-to-go: $\mathrm{Var}[\hat{g}_{\mathrm{rtg}}(\theta)] = \frac{1 + 3\theta + \theta^2}{\theta (1-\theta)^4}$",
    linewidth=2.5,
    color="orange",
)
plt.axvline(
    0.11,
    color="red",
    linestyle="--",
    alpha=0.6,
    linewidth=1.5,
    label=r"$\theta = 0.11$",
)
plt.axvline(
    0.2,
    color="red",
    linestyle="--",
    alpha=0.6,
    linewidth=1.5,
    label=r"$\theta^' = 0.2$",
)
plt.xlabel(r"$\theta$", fontsize=14)
plt.ylabel("Variance", fontsize=14)
plt.title("Variance Comparison: Original vs Return-to-Go Policy Gradient", fontsize=16)
plt.legend(fontsize=12, loc="upper right")
plt.grid(True, alpha=0.3)
plt.ylim(0, 150)  # Set reasonable y-axis limit
plt.xlim(0, 1)
plt.tight_layout()

# Save the plot
output_path = (
    "/home/siqi/Desktop/homework_fall2023-main/hw2/report/imgs/variance_comparison.png"
)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")

# Also print some statistics
print(f"\nAt theta = 0.5:")
print(f"  Original variance: {(1 + 8*0.5 + 4*0.5**2)/(0.5*(1-0.5)**4):.2f}")
print(f"  Return-to-go variance: {(1 + 3*0.5 + 0.5**2)/(0.5*(1-0.5)**4):.2f}")
