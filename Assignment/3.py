import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Step 1: Read data from the CSV file
data = pd.read_csv('weight-height.csv')

# Extract lengths and weights from the table
length = data['Weight']  # Assuming lengths are in the second column
weight = data['Height']  # Assuming weights are in the third column

# Step 2: Convert lengths and weights
length_cm = length * 2.54  # Convert inches to centimeters
weight_kg = weight * 0.453592  # Convert pounds to kilograms

# Step 3: Calculate means
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

# Print the means
print(f"Mean Length: {mean_length:.2f} cm")
print(f"Mean Weight: {mean_weight:.2f} kg")

# Step 4: Draw a histogram of the lengths
plt.hist(length_cm, bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram of Lengths (in cm)")
plt.xlabel("Length (cm)")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
