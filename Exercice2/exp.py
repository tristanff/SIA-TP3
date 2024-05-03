import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from perceptron import LinealPerceptron, HiperbolicPerceptron, LogisticPerceptron


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
error_threshold = 0.0001
max_epochs = 5000
beta = 1

learning_rate = 0.01  # Fixed learning rate
training_ratios = [0.6, 0.7, 0.8, 0.9]  # Different training ratios
num_iterations = 10  # Number of iterations to average over


# Utility function to pad arrays to the same length
def extend_to_max_length(values, max_length):
    return np.pad(values, (0, max_length - len(values)), 'edge')


def run_experiments_with_ratios(filename):
    results = []

    # Loop through each training ratio
    for train_ratio in training_ratios:
        split_index = int(len(input_data) * train_ratio)
        training_data = input_data[:split_index].tolist()  # Training data
        testing_data = input_data[split_index:].tolist()  # Testing data

        # Linear Perceptron
        all_mse = []
        max_length = 0
        for i in range(num_iterations):
            linear_perceptron = LinealPerceptron(learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = linear_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            print(f"Epochs For Lineal iteration {i} : {epochs}\n")
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Linear',
                'Training Ratio': train_ratio,
                'Epoch': epoch,
                'MSE': mse
            })

        # Hyperbolic Perceptron
        all_mse = []
        max_length = 0
        for i in range(num_iterations):
            hiperbolic_perceptron = HiperbolicPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = hiperbolic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            print(f"Epochs For Hyperbolic iteration {i} : {epochs}\n")
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Hyperbolic',
                'Training Ratio': train_ratio,
                'Epoch': epoch,
                'MSE': mse
            })

        # Logistic Perceptron
        all_mse = []
        max_length = 0
        for i in range(num_iterations):
            logistic_perceptron = LogisticPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = logistic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            print(f"Epochs For Logistic iteration {i} : {epochs}\n")
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Logistic',
                'Training Ratio': train_ratio,
                'Epoch': epoch,
                'MSE': mse
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)  # Save to CSV


# Save data to CSV with fixed learning rate
csv_filename = 'MSE_byTrainRatio-Epochs.csv'
run_experiments_with_ratios(csv_filename)