import numpy as np
import random
import math
from abc import ABC, abstractmethod

from typing import List, Tuple


# Abstract base class for different types of perceptrons
class Perceptron(ABC):
    def __init__(self, learning_rate, weights, bias, input_data_dimension, epsilon):
        # If no weights provided, initialize with random values; otherwise, use provided weights
        if weights is None or len(weights) == 0:
            self.weights = np.random.rand(input_data_dimension + 1)  # Include bias term (w_0)
        else:
            self.weights = weights
        # Store other parameters used for training and computation
        self.learning_rate = learning_rate
        self.bias = bias
        self.epsilon = epsilon

    @abstractmethod
    def activation_function(self, x):
        # This function must be defined by subclasses
        pass

    @abstractmethod
    def activation_derivative(self, x):
        # This derivative function must also be defined by subclasses
        pass

    def compute_error(self, input_data, expected):
        # Calculate the Mean Squared Error between expected outputs and current outputs
        outputs = self.current_outputs(input_data)

        total_error = 0
        # Accumulate the squared differences for each data point
        for mu in range(len(expected)):
            scaled_expected = self.activation_function(expected[mu])  # Scale the expected output
            total_error += (scaled_expected - outputs[mu]) ** 2
        # Return the mean error
        return total_error / len(expected)

    def theta(self, x):
        # Add a bias term to the input vector and compute dot product with weights
        extended_x = np.array([1] + x)
        return np.dot(extended_x, self.weights) + self.bias  # Compute linear combination with weights

    def current_outputs(self, input_data) -> List[float]:
        # Compute outputs for all input data using the activation function
        return [self.activation_function(self.theta(x_mu)) for x_mu in input_data]

    def delta_weights(self, excitement, activation, expected, x_mu):
        # Compute the weight update based on the difference between expected and actual output
        extended_x_mu = np.array([1] + x_mu)  # Include bias term
        scaled_expected = self.activation_function(expected)  # Scale expected output
        # Compute the change in weights based on the error and activation derivative
        return self.learning_rate * (scaled_expected - activation) * self.activation_derivative(
            excitement) * extended_x_mu

    def train(self, input_data, testing_data, expected, expected_test, epoch_max, scale=False):
        epoch = 0  # Keep track of the current epoch
        error_min = math.inf  # Start with infinite error
        input_len = len(input_data)  # Number of training samples

        # If scaling is enabled, normalize expected outputs
        if scale:
            expected = [self.scale_result(value, min(expected), max(expected)) for value in expected]
            expected_test = [self.scale_result(value, min(expected_test), max(expected_test)) for value in
                             expected_test]

        # To store error values during training for analysis
        train_errors = []
        test_errors = []

        # Training loop, runs until error is below epsilon or epoch limit is reached
        while error_min > self.epsilon and epoch < epoch_max:
            # Randomly select a training sample
            mu = random.randrange(0, input_len)
            x_mu = input_data[mu]
            excitement = self.theta(x_mu)  # Linear combination with current weights
            activation = self.activation_function(excitement)  # Apply activation function

            # Update weights based on the error
            self.weights += self.delta_weights(excitement, activation, expected[mu], x_mu)

            # Calculate the new training error
            new_error = self.compute_error(input_data, expected)

            # Store errors for future analysis
            train_errors.append(new_error)
            test_errors.append(self.compute_error(testing_data, expected_test))

            # Update minimum error found so far
            if new_error < error_min:
                error_min = new_error

            epoch += 1  # Increment epoch counter

        # Return total epochs and the error progression for both training and testing
        return epoch, train_errors, test_errors

    def scale_result(self, value, new_min, new_max):
        # Normalize a value to a new range
        scaled = ((value - new_min) / (new_max - new_min)) * (
                self.activation_max - self.activation_min) + self.activation_min
        return scaled

    def _predict_one(self, x: List[int], scale: bool, scale_interval: Tuple[float]):
        excitement = self.theta(x)  # Compute linear combination with weights
        activation = self.activation_function(excitement)  # Apply activation function
        return activation  # Return the final prediction

    def predict(self, input_data, scale: bool = False, scale_interval: Tuple[float] = ()) -> \
            Tuple[List[float], float]:
        # If scaling is required, ensure valid scale_interval is provided
        if scale:
            if scale_interval is None or len(scale_interval) != 2:
                raise Exception("Wrong scale_interval: size must be 2")
            if scale_interval[0] > scale_interval[1]:
                raise Exception("Wrong scale_interval: min > max")

        result = []
        # Generate predictions for all input data
        for x in input_data:
            result.append(self._predict_one(x, scale=scale, scale_interval=scale_interval))

        # Calculate the Mean Squared Error of predictions
        test_mse = self.compute_error(input_data, result)

        return result, test_mse  # Return predictions and their MSE


# Perceptron with a linear activation function
class LinealPerceptron(Perceptron):
    def activation_function(self, x):
        # The activation function is identity (linear)
        return x

    def activation_derivative(self, x):
        # Derivative of a linear function is constant 1
        return 1


# Perceptron with a hyperbolic tangent activation function
class HiperbolicPerceptron(Perceptron):
    def __init__(self, beta, learning_rate, weights, bias, input_data_dimension, epsilon):
        # Set beta and the activation range for tanh
        self.beta = beta
        self.activation_min = -1.0
        self.activation_max = 1.0
        super().__init__(learning_rate, weights, bias, input_data_dimension, epsilon)

    def activation_function(self, x):
        # Hyperbolic tangent, output between -1 and 1
        return math.tanh(self.beta * x)

    def activation_derivative(self, x):
        # Derivative for tanh function
        return self.beta * (1 - self.activation_function(x) ** 2)


# Perceptron with a logistic activation function (sigmoid)
class LogisticPerceptron(Perceptron):
    def __init__(self, beta, learning_rate, weights, bias, input_data_dimension, epsilon):
        # Set beta and the activation range for logistic function
        self.beta = beta
        self.activation_min = 0.0
        self.activation_max = 1.0
        super().__init__(learning_rate, weights, bias, input_data_dimension, epsilon)

    def activation_function(self, x):
        # Logistic (sigmoid) function, output between 0 and 1
        return 1 / (1 + math.exp(-self.beta * x))

    def activation_derivative(self, x):
        # Derivative for logistic function, used in training
        return 2 * self.beta * self.activation_function(x) * (1 - self.activation_function(x))
