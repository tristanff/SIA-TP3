import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron import LinealPerceptron, HiperbolicPerceptron, LogisticPerceptron
from collections import defaultdict

# Helper function to scale values between a minimum and a maximum
def scale_result(value, min_val, max_val):
    if min_val == max_val:
        return 0
    return (value - min_val) / (max_val - min_val)

# Function to determine the range for scaling
def min_max_interval(values):
    return min(values), max(values)

# Define parameters
perceptron_types = ["Hyperbolic", "Logistic", "Linear"]
learning_rate = 0.01
training_percentage = 0.8  # 80% for training, 20% for testing
max_epochs = 10000
bias = 1
beta = 1
error_threshold = 0.001  # Error threshold

# Function to get data and split into training and testing sets
def get_data():
    data = pd.read_csv('TP3-ej2-conjunto.csv')  # Change this with your CSV file
    input_data = np.array(data[['x1', 'x2', 'x3']])  # Input variables
    expected_data = np.array(data['y'])  # Expected outputs
    return input_data, expected_data

# Initialize the data
input_data, expected_data = get_data()
total_data_count = len(input_data)

training_count = int(total_data_count * training_percentage)
training_set = input_data[:training_count].tolist()
testing_set = input_data[training_count:].tolist()

expected_train = expected_data[:training_count]
expected_test = expected_data[training_count:]

# Function to initialize perceptrons based on type
def initialize_perceptron(perceptron_type, beta, learning_rate, bias, input_size, error_threshold):
    if perceptron_type == "Linear":
        return LinealPerceptron(learning_rate, [], bias, input_size, error_threshold)
    elif perceptron_type == "Hyperbolic":
        return HiperbolicPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)
    elif perceptron_type == "Logistic":
        return LogisticPerceptron(beta, learning_rate, [], bias, input_size, error_threshold)

# Set the range of epsilon values
epsilons = np.arange(0.5, 0.0, -0.1)

# Define a dictionary to store the average error rates for each perceptron type and epsilon
average_error_rates = defaultdict(lambda: defaultdict(list))

# Repeat the experiment 10 times
num_iterations = 10

for iteration in range(num_iterations):
    print(f"\nIteration {iteration + 1}/{num_iterations}")
    for epsilon in epsilons:
        print(f"Testing with epsilon: {epsilon:.1f}")
        for perceptron_type in perceptron_types:
            perceptron = initialize_perceptron(perceptron_type, beta, learning_rate, bias, 3, error_threshold)

            if perceptron_type == "Linear":
                _, train_errors, test_errors = perceptron.train(
                    training_set, testing_set, expected_train, expected_test, max_epochs
                )
                results, test_mse = perceptron.predict(testing_set)
            else:
                _, train_errors, test_errors = perceptron.train(
                    training_set, testing_set, expected_train, expected_test, max_epochs, scale=True
                )
                results, test_mse = perceptron.predict(
                    testing_set, scale=True, scale_interval=min_max_interval(expected_data)
                )

            correct_predictions = 0
            min_val, max_val = min_max_interval(expected_test)

            for i in range(len(results)):
                expected_scaled = scale_result(expected_test[i], min_val, max_val) if perceptron_type != "Linear" else expected_test[i]
                result_value = results[i]
                delta = abs(result_value - expected_scaled)

                if delta <= epsilon:
                    correct_predictions += 1

            total_predictions = len(results)
            error_rate = (total_predictions - correct_predictions) / total_predictions * 100

            # Store the error rate for each perceptron type and epsilon in the dictionary
            average_error_rates[perceptron_type][epsilon].append(error_rate)

            # Output summary
            print(f"{perceptron_type} Perceptron with epsilon={epsilon:.1f}")
            print(f"Correct Predictions: {correct_predictions} out of {total_predictions}")
            print(f"Training Error: {train_errors[-1]}")
            print(f"Testing Error: {test_mse}")
            print(f"Error Rate: {error_rate:.2f}%\n")

# Now calculate the average error rates for each perceptron type and epsilon
final_average_error_rates = {}

for perceptron_type in perceptron_types:
    final_average_error_rates[perceptron_type] = {
        epsilon: np.mean(average_error_rates[perceptron_type][epsilon])
        for epsilon in epsilons
    }

# Create a plot to display the average error rates for each perceptron type and epsilon
for perceptron_type in perceptron_types:
    plt.plot(list(final_average_error_rates[perceptron_type].keys()),
             list(final_average_error_rates[perceptron_type].values()),
             label=perceptron_type)

plt.xlabel("Epsilon")
plt.ylabel("Average Error Rate (%)")
plt.title("Average Error Rate by Perceptron Type and Epsilon (10 Iterations)")
plt.legend()
plt.show()
