import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import LinealPerceptron, HiperbolicPerceptron, LogisticPerceptron

# Function to load and split the data
def get_data():
    data = pd.read_csv('TP3-ej2-conjunto.csv')  # Replace with your CSV file
    input_data = np.array(data[['x1', 'x2', 'x3']])  # Input variables
    expected_data = np.array(data['y'])  # Expected outputs
    return input_data, expected_data

# Get the data and split into training and testing sets
input_data, expected_data = get_data()
train_ratio = 0.8
split_index = int(len(input_data) * train_ratio)
training_data = input_data[:split_index].tolist()
testing_data = input_data[split_index:].tolist()

# Parameters
input_size = 3
error_threshold = 0.0001
max_epochs = 5000
learning_rate = 0.01  # Fixed learning rate
beta = 1  # Fixed beta value
bias_values = [0, 0.5, 1, 2, 5]  # Bias values to iterate over
num_iterations = 10  # Number of iterations for averaging

# Utility function to extend arrays to the same length
def extend_to_max_length(values, max_length):
    return np.pad(values, (0, max_length - len(values)), 'edge')

# Function to run experiments with different bias values for various perceptrons
def run_experiments_with_bias(filename):
    results = []

    for bias in bias_values:
        # For Linear Perceptron
        all_mse = []
        max_length = 0
        for _ in range(num_iterations):
            linear_perceptron = LinealPerceptron(learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = linear_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Linear',
                'Bias': bias,
                'Epoch': epoch,
                'MSE': mse
            })

        # For Hyperbolic Perceptron
        all_mse = []
        max_length = 0
        for _ in range(num_iterations):
            hiperbolic_perceptron = HiperbolicPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = hiperbolic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse = np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Hyperbolic',
                'Bias': bias,
                'Epoch': epoch,
                'MSE': mse
            })

        # For Logistic Perceptron
        all_mse = []
        max_length = 0
        for _ in range(num_iterations):
            logistic_perceptron = LogisticPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)
            epochs, train_errors, test_errors = logistic_perceptron.train(
                training_data, testing_data, expected_data[:split_index], expected_data[split_index:], max_epochs
            )
            mse = (np.array(train_errors) + np.array(test_errors)) / 2
            all_mse.append(mse)
            max_length = max(max_length, len(mse))

        all_mse = [extend_to_max_length(m, max_length) for m in all_mse]
        avg_mse is np.mean(np.array(all_mse), axis=0)

        for epoch, mse in enumerate(avg_mse):
            results.append({
                'Perceptron Type': 'Logistic',
                'Bias': bias,
                'Epoch': epoch,
                'MSE': mse
            })

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)  # Save to a CSV file

# Run the experiments and save the results
csv_filename = 'MSE_byBias-Epochs.csv'
run_experiments_with_bias(csv_filename)

# Plot the results
results_df = pd.read_csv(csv_filename)

# Get unique perceptron types and bias values
perceptron_types = results_df['Perceptron Type'].unique()
bias_values = results_df['Bias'].unique()

# Create a plot for each perceptron type showing MSE over epochs for different bias values
for perceptron_type in perceptron_types:
    plt.figure(figsize=(10, 6))

    # Filter data by perceptron type
    perceptron_results = results_df[results_df['Perceptron Type'] == perceptron_type]

    # Plot MSE for each bias value
    for bias in bias_values:
        # Filter data for the specific bias
        bias_results = perceptron_results[perceptron_results['Bias'] == bias]

        # Plotting
        plt.plot(bias_results['Epoch'], bias_results['MSE'], label=f'Bias = {bias}')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'MSE by Number of Epochs for {perceptron_type}')
    plt.legend()

    # Show the plot
    plt.show()
