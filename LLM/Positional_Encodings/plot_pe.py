import numpy as np
import matplotlib.pyplot as plt

# Function to generate absolute positional encoding (heatmap)
def plot_absolute_encoding():
    positions = np.arange(10)
    dimensions = np.arange(16)
    encoding = np.array([[np.sin(pos / (10000 ** (2 * i / 16))) if i % 2 == 0 
                          else np.cos(pos / (10000 ** (2 * (i-1) / 16))) 
                          for i in dimensions] for pos in positions])

    plt.figure(figsize=(8, 4))
    plt.imshow(encoding, aspect='auto', cmap='viridis')
    plt.colorbar(label="Encoding Value")
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Token Position")
    plt.title("Absolute Positional Encoding (APE)")
    plt.show()

# Function to generate relative positional encoding (distance matrix)
def plot_relative_encoding():
    positions = np.arange(10)
    relative_encoding = np.abs(positions[:, None] - positions[None, :])  # Distance matrix

    plt.figure(figsize=(6, 5))
    plt.imshow(relative_encoding, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label="Relative Distance")
    plt.xlabel("Token Position")
    plt.ylabel("Token Position")
    plt.title("Relative Positional Encoding (RPE)")
    plt.show()

# Function to generate rotary positional encoding (rotation visualization)
def plot_rotary_encoding():
    positions = np.arange(10)
    angles = positions * (np.pi / 10)  # Rotation angles based on position

    plt.figure(figsize=(5, 5))
    for i, angle in enumerate(angles):
        x, y = np.cos(angle), np.sin(angle)
        plt.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, color=plt.cm.viridis(i / len(angles)))

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("Rotary Positional Encoding (RoPE)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

# Generate all three visualizations
plot_absolute_encoding()
plot_relative_encoding()
plot_rotary_encoding()
