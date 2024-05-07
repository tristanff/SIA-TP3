# Created separate file to keep it simple. 
# Beware of long compute time with high epoch number and many learning rates

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

class NeuralNetwork:
    def __init__(self, layer_sizes, one_hot=False, activation_function='sigmoid', optimizer='adam'):
        # Initialize number of nodes of input, hidden, and output layer
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Use of one_hot encoding (True/False)
        self.one_hot = one_hot

        # Activation function
        self.activation_function = activation_function

        # Optimizer setting
        self.optimizer = optimizer

        # Initialize weights (w) and biases (b)
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        
        # Record data
        self.losses = []

    def activation(self, x, deriv=False):
        """Activation function (sigmoid, reLU)"""
        if self.activation_function == 'sigmoid':
            if deriv:
                return self.activation(x) * (1 - self.activation(x))
            return 1 / (1 + np.exp(-x))
        
        elif self.activation_function == 'reLU':
            if deriv:
                return 1 * (x > 0)
            return np.maximum(0, x)

    def one_hot_encoding(self, y):
        """Encode digit to one-hot (used in error calculation of backpropagation)"""
        y = y.reshape(-1, 10)
        encoded_y = np.zeros((y.size, y.max() + 1))
        encoded_y[np.arange(y.size), y.flatten()] = 1
        encoded_y = encoded_y.T
        return encoded_y

    def feedforward(self, x):
        """Feedforward"""
        self.zs = []
        self.activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            a = self.activation(z)
            self.zs.append(z)
            self.activations.append(a)
        return a

    def backpropagation(self, x, y, learning_rate):
        """Backpropagation"""
        m = y.size
    
        # Backward pass
        d_z = self.activations[-1] - y
        for i in range(len(self.weights) - 1, -1, -1):
            d_w = np.dot(self.activations[i].T, d_z) / m
            d_b = np.sum(d_z) / m
    
            self.weights[i] -= d_w * learning_rate
            self.biases[i] -= d_b * learning_rate
    
            if i > 0:
                d_z = np.dot(d_z, self.weights[i].T) * self.activation(self.zs[i - 1], deriv=True)

    def train(self, x, y, learning_rate, epochs):
        """Optimize weight and bias parameters"""
        self.input = x
        self.output = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        all_accuracies = []  # Store accuracies for each epoch

        if self.one_hot:
            y = self.one_hot_encoding(y)

        #x = (x - np.mean(x)) / np.std(x)
        
        for epoch in range(epochs):
            self.feedforward(x)
            self.backpropagation(x, y, learning_rate)

            # Calculate accuracy
            accuracy = self.calculate_accuracy(x, y)
            all_accuracies.append(accuracy) 

            loss = np.mean(np.square(y - self.activations[-1]))
            self.losses.append(loss)

            if not epoch % (epochs / 10):
                print(f'Epoch {epoch}: {loss}')
            
        return all_accuracies  # Return the accuracies

    def calculate_accuracy(self, x, y):
        predictions = self.predict(x)
        #if self.one_hot:  # Adapt if one-hot encoding is used
        #    y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy 

    def predict(self, x):
        """Get network output with current parameters"""
        prediction = self.feedforward(x)
        if self.layer_sizes[-1] == 1:
            return (prediction > 0.5).astype(int)
        return self.output[np.argmax(prediction, axis=1)]

    def graph(self, graph_type, name=None, xlim=None):
        """Graph information about neural network performance"""
        
        # Check graph type
        if graph_type == None:
            raise ValueError('Unknown graph type')
        
        # Generate name if no name is given
        if name == None:
            name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

        # Add parameter box for plot
        parameters = (
            'Parameters\n\n'
            f'Layers: {self.num_layers}\n'
            f'Input nodes: {self.layer_sizes[0]}\n'
            f'Hidden nodes: {self.layer_sizes[1:-1]}\n'
            f'Output nodes: {self.layer_sizes[-1]}\n'
            f'Learning rate: {self.learning_rate}\n'
            f'Epochs: {self.epochs}'
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [8, 2]}, figsize=(8.5, 4.8))
        ax2.text(-0.2, 0.7, parameters, bbox=dict(facecolor='white', alpha=0.5), transform=ax2.transAxes)
        ax2.axis('off')

        # Set x-axis limits
        if xlim != None:
            ax1.set_xlim(xlim)

        # Plot loss vs epoch
        if graph_type == 'loss':
            ax1.plot(self.losses)            
            ax1.set_title('Loss vs epoch')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid()
            fig.savefig(f'./graphs/loss_{name}', dpi=300)

        # Plot confusion matrix
        elif graph_type == 'confusion':
            matrix = np.array([self.feedforward(value).squeeze() for value in self.input]).T      
            im = ax1.matshow(matrix, cmap='viridis');

            for i in range(matrix.shape[0]):                
                for j in range(matrix.shape[1]):
                    ax1.text(j, i, f'{np.round(matrix[i, j], 2)}', ha='center', va='center', color='white', size='8')

            ax1.set_title('Confusion matrix')
            ax1.set_xlabel('Predicted digit')
            ax1.set_ylabel('Real digit')
            ax1.set_xticks(np.arange(len(self.input)))
            ax1.set_yticks(np.arange(len(self.input)))
            fig.colorbar(im, ax=ax1)
            fig.savefig(f'./graphs/confusion_{name}', dpi=300);
           
####### PARAMETERS ########    
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

epochs = 5000

learning_rates = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1]

nn = NeuralNetwork([2,2,1]) 
    

######## Plot accuracy vs epochs for each learning rate
all_learning_rates_accuracies = {}  # A dictionary to store results

for learning_rate in learning_rates:
    accuracies = nn.train(x, y, learning_rate, epochs)
    all_learning_rates_accuracies[learning_rate] = accuracies


plt.figure(figsize=(8, 5))
fig, ax = plt.subplots()  # Create a single plot

for learning_rate, accuracies in all_learning_rates_accuracies.items():
    plt.plot(all_learning_rates_accuracies[learning_rate])  # Plot accuracies
    ax.plot(accuracies, label=f'Learning Rate: {learning_rate}')
    
## Generate plot
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs. Epochs (Multiple Learning Rates)')
ax.grid(True)
ax.legend() 
plt.ylim(0,1)
plt.show()
