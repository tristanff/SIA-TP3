import numpy as np
import matplotlib.pyplot as plt


# Activation function (Unit Step Function)
def f(z):
    return 1 if z >= 0 else 0


# Simple Perceptron
def perceptronModel(x, weights, bias):
    z = np.dot(weights, x) + bias  # Combination of weights and inputs with bias
    y = f(z)  # Apply activation function
    return y


# Data for training (AND logic function)
x = np.array([
    [0, 1],  # Input 1
    [1, 1],  # Input 2
    [0, 0],  # Input 3
    [1, 0]  # Input 4
])

y = np.array([0, 1, 0, 0])  # Expected outputs



def perceptronSimple_algo(x, y, num_iterations, weights, bias, learning_rate):
    for _ in range(num_iterations):
        for xi, yi in zip(x, y):
            prediction = perceptronModel(xi, weights, bias)
            error = yi - prediction

            weights += learning_rate * error * xi
            bias += learning_rate * error

    # Return the weights and bias as a dictionary
    result = {
        "weights": weights,
        "bias": bias
    }
    return result


weights = np.array([0.0, 0.0])  # Initial weights
bias = 0.0  # Initial bias
learning_rate = 0.1
num_iterations = 10

# Train the perceptron
result = perceptronSimple_algo(x, y, num_iterations, weights, bias, learning_rate)

# Display the found weights and bias
print("Found weights: w1 = {:.2f}, w2 = {:.2f}".format(result["weights"][0], result["weights"][1]))
print("Found bias: b = {:.2f}".format(result["bias"]))

# Plot the data points for the AND problem
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', s=100, label='Data points')

# Plot the decision line
x1_vals = np.array([-0.5, 1.5])  # Range of values for the x1 axis
x2_vals = -(result["weights"][0] * x1_vals + result["bias"]) / result["weights"][1]  # Compute the decision line
plt.plot(x1_vals, x2_vals, 'k-', label='Decision line')

# Chart parameters
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Simple Perceptron - AND Logic Function')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


# XOR Training Data
x = np.array([
    [0, 1],  # Entrée 1
    [1, 1],  # Entrée 2
    [0, 0],  # Entrée 3
    [1, 0]   # Entrée 4
])


## XOR Function ##
y = np.array([1, 0, 0, 1])  # Expected outplus

weights = np.array([0.0, 0.0])  # Initial weights
bias = 0.0  # Initial bias
learning_rate = 0.1
num_iterations = 10


result = perceptronSimple_algo(x, y, num_iterations, weights, bias, learning_rate)

# Affichage des poids et du biais trouvés
print("Found weights: w1 = {:.2f}, w2 = {:.2f}".format(result["weights"][0], result["weights"][1]))
print("Found bias: b = {:.2f}".format(result["bias"]))


# Tracé des points du problème XOR
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', s=100, label='Data points')

# Tracé de la ligne de séparation
x1_vals = np.array([-0.5, 1.5])  # Plage de valeurs pour l'axe x1
x2_vals = -(result["weights"][0] * x1_vals + result["bias"]) / result["weights"][1]  # Calcule la ligne de séparation
plt.plot(x1_vals, x2_vals, 'k-', label='Decision line')

# Paramètres du graphique
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Simple Perceptron - XOR Logic Function')
plt.legend()
plt.grid(True)

# Affichage du graphique
plt.show()
