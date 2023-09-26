import numpy as np
import math

n = 13      # size of hilbert matrix
num_multiplications = 0

# Create the Hilbert matrix
hilbert_matrix = np.zeros((n, n))
for i in range(1, n + 1):
    for j in range(1, n + 1):
        hilbert_matrix[i - 1, j - 1] = 1.0 / (i + j - 1)
        num_multiplications += 1

# augment matrix with all 1's
augmented_matrix = np.column_stack((hilbert_matrix, np.ones(n)))
augmented_matrix = np.round(augmented_matrix, 4) # keep 4 digit rounding property

# Perform Gaussian elimination
for i in range(n):
    # Eliminate entries below the current pivot
    for j in range(i + 1, n):
        pivot = augmented_matrix[i, i]
        multiplier = augmented_matrix[j, i] / pivot
        augmented_matrix[j, i:] -= multiplier * augmented_matrix[i, i:]
        num_multiplications += 2
        
    # Round the values in the augmented matrix
    augmented_matrix = np.round(augmented_matrix, 4)


# Backward substitution
x = np.zeros(n)     # initialize solution vector x
# loop from n-1 -> 0, decremeting by 1 every step
for i in range(n - 1, -1, -1):
    sum = 0
    for j in range(n, i, -1):
        if (j != n):
            sum -= augmented_matrix[i, j] * x[j]
            num_multiplications += 1
        # first loop through special case (augmented column)
        else:
            sum += augmented_matrix[i, j]
    # whatshere = augmented_matrix[i,i]
    x[i] = sum / augmented_matrix[i, i]
    num_multiplications += 1
# Round the values in the x vector
x = np.round(x, 4)

# Error Vector
error_vector = np.subtract(np.ones(n), x)
error_vector = np.round(error_vector, 4)

# Euclidean Norm
euclidean_norm = 0
for i in x:
    euclidean_norm += i * i
euclidean_norm = math.sqrt(euclidean_norm)
euclidean_norm = round(euclidean_norm, 4)

# Infinity Norm
absolute_X = abs(x)
infinity_norm = np.max(absolute_X)

# Print DATA
print("Solution Vector:")
print(x)
print("Error Vector:")
print(error_vector)
print(f"Number of Multiplications: {num_multiplications}")
print(f"Euclidean Norm: {euclidean_norm}")
print(f"Infinity Norm: {infinity_norm}")
