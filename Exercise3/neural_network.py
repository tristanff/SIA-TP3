import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

class NeuralNetwork:
    def __init__(self, layer_sizes, one_hot=False, activation_function='sigmoid', optimizer='gradient'):
        # Initialize number of nodes of input, hidden, and output layer
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Use of one_hot encoding (True/False)
        self.one_hot = one_hot

        # Activation function
        self.activation_function = activation_function

        # Initialize weights (w) and biases (b)
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.num_layers - 1)]

        # Optimizer setting
        self.optimizer = optimizer
        
        # Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.t = 0     
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
        y = y.reshape(-1, len(y))
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
        """Backpropagation with chosen optimizer"""
        if self.optimizer == 'adam':
            self.adam_optimizer(x, y, learning_rate)
        else:
            self.gradient_optimizer(x, y, learning_rate)
    
    def adam_optimizer(self, x, y, learning_rate):
        """Backpropagation with Adam optimizer"""
        m = y.size
        self.t += 1
        
        # Ensure initialization of m and v
        if self.t == 1:
            self.m = [np.zeros_like(w) for w in self.weights]
            self.v = [np.zeros_like(w) for w in self.weights]
        
        # Backward pass
        d_z = self.activations[-1] - y
        for i in range(len(self.weights) - 1, -1, -1):
            d_w = np.dot(self.activations[i].T, d_z) / m
            d_b = np.sum(d_z) / m
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * d_w
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (d_w ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update weights and biases
            self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.biases[i] -= learning_rate * d_b
    
            if i > 0:
                d_z = np.dot(d_z, self.weights[i].T) * self.activation(self.zs[i - 1], deriv=True)

    def gradient_optimizer(self, x, y, learning_rate):
        """Backpropagation with gradient descent"""
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

    def train(self, x, y, learning_rate, epochs, verbose=False):
        """Optimize weight and bias parameters"""
        self.input = x
        self.output = y
        self.learning_rate = learning_rate
        self.epochs = epochs

        if self.one_hot:
            y = self.one_hot_encoding(y)

        #x = (x - np.mean(x)) / np.std(x)
        
        for epoch in range(epochs):
            self.feedforward(x)
            self.backpropagation(x, y, learning_rate)

            loss = np.mean(np.square(y - self.activations[-1]))
            self.losses.append(loss)

            if verbose and not epoch % (epochs / 10):
                print(f'Epoch {epoch}: {loss}')

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
            f'Activation function: {self.activation_function}\n'
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