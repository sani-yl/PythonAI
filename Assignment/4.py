import numpy as np

# Define matrix A
A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

# Compute products AA^(-1) and A^(-1)A
identity1 = np.dot(A, A_inv)
identity2 = np.dot(A_inv, A)

# Print results
print("Matrix A:")
print(A)

print("\nInverse of Matrix A (A^-1):")
print(A_inv)

print("\nProduct AA^-1:")
print(identity1)

print("\nProduct A^-1A:")
print(identity2)

# Check if they are close to the unit matrix
is_identity1 = np.allclose(identity1, np.eye(3))
is_identity2 = np.allclose(identity2, np.eye(3))

print(f"\nAA^-1 is close to the unit matrix: {is_identity1}")
print(f"A^-1A is close to the unit matrix: {is_identity2}")
