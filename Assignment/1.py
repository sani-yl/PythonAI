import numpy as np
import matplotlib.pyplot as plt

# Define the range for x values
x = np.linspace(-10, 10, 500)

# Define the lines
y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

# Plot the lines with different styles and labels
plt.plot(x, y1, label="y=2x+1", color="green", linestyle="-",)
plt.plot(x, y2, label="y=2x+2", color="blue", linestyle="--",)
plt.plot(x, y3, label="y=2x+3", color="red", linestyle="-.")

# Set the title and labels
plt.title("Graphs of y=2x+1, y=2x+2, and y=2x+3")
plt.xlabel("x")
plt.ylabel("y")

# Add a legend to distinguish the lines
plt.legend()

# Add a grid for better visualization
plt.grid(True, linestyle="-.")

# Display the plot
plt.show()

