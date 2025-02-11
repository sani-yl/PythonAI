import numpy as np
import matplotlib.pyplot as plt

# Define the values of 'n'
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

# 1) Possible sums when throwing two dice
possible_sums = np.arange(2, 13)

# Loop over each value of n
for n in n_values:
    # simulate throwing two dice n times
    dice1 = np.random.randint(1, 7, size = n)
    dice2 = np.random.randint(1, 7, size = n)
    sums = dice1 + dice2
    
    # 2) Compute the histogram
    h, h2 = np.histogram(sums, bins=range(2, 14))
    
    # 3) Plotting the histogram
    plt.bar(h2[:-1], h / n, width=0.8, alpha=0.7)
    plt.title(f'Histogram of Sums of Two Dice Rolls (n={n})')
    plt.xlabel('Sum of Dice')
    plt.ylabel('Frequency')
    plt.xticks(range(2, 13))
    plt.show()