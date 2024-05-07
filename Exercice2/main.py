import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from perceptron import LinealPerceptron, HiperbolicPerceptron, LogisticPerceptron

# Function to get input and expected data from a CSV file
def get_data():
    data = pd.read_csv('TP3-ej2-conjunto.csv')  # Change this to your CSV file
    input_data = np.array(data[['x1', 'x2', 'x3']])  # Input features
    expected_data = np.array(data['y'])  # Expected outputs
    return input_data, expected_data

input_data, expected_data = get_data()

# Define a split ratio for training and testing data
train_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(len(input_data) * train_ratio)

training_data = input_data[:split_index].tolist()  # Training data
testing_data = input_data[split_index:].tolist()  # Testing data

# Hyperparameters
input_size = 3
bias = 1
error_threshold = 0.01
max_epochs = 5000
beta = 1  # Default value for beta


learning_rates = [0.0001, 0.001, 0.01]

num_iterations = 10

# Utility function to pad arrays to the same length
def extend_to_max_length(values, max_length):
    return np.pad(values, (0, max_length - len(values)), 'edge')

# Create a figure with three subplots
plt.figure(figsize=(15, 5))

# Plot for the Linear Perceptron
plt.subplot(1, 3, 1)
for lr in learning_rates:
    all_mse = []  # List to store MSE for all iterations
    max_length = 0  # To track the longest MSE sequence
    for _ in range(num_iterations):
        linear_perceptron = LinealPerceptron(lr, [], bias, input_size, error_threshold)
        epochs, train_errors, test_errors = linear_perceptron.train(
            training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs,scale=True
        )
        mse = (np.array(train_errors) + np.array(test_errors)) / 2
        all_mse.append(mse)
        max_length = max(max_length, len(mse))  # Track the maximum length

    # Extend all MSE to the maximum length
    all_mse = [extend_to_max_length(m, max_length) for m in all_mse]

    # Calculate the average MSE
    avg_mse = np.mean(np.array(all_mse), axis=0)
    plt.plot(range(len(avg_mse)), avg_mse, label=f'LR = {lr}')  # Plot the average MSE
plt.xlabel("Number of epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Linear Perceptron")
plt.legend()
plt.grid(True)

# Plot for the Hyperbolic Perceptron
plt.subplot(1, 3, 2)
for lr in learning_rates:
    all_mse = []
    max_length = 0
    for _ in range(num_iterations):
        hiperbolic_perceptron = HiperbolicPerceptron(beta, lr, [], bias, input_size, error_threshold)
        epochs, train_errors, test_errors = hiperbolic_perceptron.train(
            training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
        )
        mse = (np.array(train_errors) + np.array(test_errors)) / 2
        all_mse.append(mse)
        max_length = max(max_length, len(mse))

    all_mse = [extend_to_max_length(m, max_length) for m in all_mse]

    avg_mse = np.mean(np.array(all_mse), axis=0)  # Calculate the average MSE
    plt.plot(range(len(avg_mse)), avg_mse, label=f'LR = {lr}')  # Plot the average MSE
plt.xlabel("Number of epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Hyperbolic Perceptron")
plt.legend()
plt.grid(True)

# Plot for the Logistic Perceptron
plt.subplot(1, 3, 3)
for lr in learning_rates:
    all_mse = []
    max_length = 0
    for _ in range(num_iterations):
        logistic_perceptron = LogisticPerceptron(beta, lr, [], bias, input_size, error_threshold)
        epochs, train_errors, test_errors = logistic_perceptron.train(
            training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
        )
        mse = (np.array(train_errors) + np.array(test_errors)) / 2
        all_mse.append(mse)
        max_length = max(max_length, len(mse))  # Track the maximum length

    all_mse = [extend_to_max_length(m, max_length) for m in all_mse]

    avg_mse = np.mean(np.array(all_mse), axis=0)  # Calculate the average MSE
    plt.plot(range(len(avg_mse)), avg_mse, label=f'LR = {lr}')  
plt.xlabel("Number of epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Logistic Perceptron")
plt.legend()
plt.grid(True)


plt.tight_layout()  
plt.show()  

