from typing import List
import numpy as np
import math


def sigmoid(val):
    return 1 / (1 + pow(math.e, -val))


class Neuron:
    def __init__(self, weights, bias=-1, learning_rate=0.1):
        self.weights = weights
        self.weights.append(bias)
        self.learning_rate = learning_rate
        self.inputs = []  # Add an attribute to store inputs

    def activation(self, cursum):
        return sigmoid(cursum)

    def output(self, inputs: List[float]):
        self.inputs = inputs + [-1]  # Store the inputs
        cursum = 0
        for i in range(0, len(self.inputs)):
            cursum += self.inputs[i] * self.weights[i]
        return self.activation(cursum)

    def update(self, inputs: List[float], desired_output):
        # Get the actual output
        actual_output = self.output(inputs)

        # Calculate error
        error = desired_output - actual_output

        # Update each weight
        for i in range(len(self.weights)):
            # For bias weight, the input is -1
            input_val = -1 if i == len(self.weights) - 1 else inputs[i]

            # Delta Rule: Adjust weights
            self.weights[i] += self.learning_rate * error * input_val


class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron([np.random.randn() for _ in range(
            input_size)], np.random.randn()) for _ in range(num_neurons)]
        self.output = None

    def feedforward(self, inputs):
        self.output = [neuron.output(inputs) for neuron in self.neurons]
        return self.output


class NeuralNetwork:
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def feedforward(self, input_data):
        for layer in self.layers:
            input_data = layer.feedforward(input_data)
        return input_data

    def backpropagate(self, X, y, learning_rate):
        # Forward pass
        X = np.array(X)
        y = np.array(y)
        output = self.feedforward(X)

        # Calculate error (assuming binary cross-entropy loss for binary classification)
        error = -(y - output)

        # Backward pass: propagate errors back and update weights
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = error if i == len(self.layers) - 1 else next_layer_errors
            next_layer_errors = []

            for j, neuron in enumerate(layer.neurons):
                for k in range(len(neuron.weights)):
                    # Calculate delta
                    delta = errors[j] * \
                        neuron.activation(neuron.inputs[k])

                    # Update weights
                    neuron.weights[k] -= learning_rate * delta

                    # Calculate error for the next layer
                    if i != 0:
                        next_layer_errors.append(delta * neuron.weights[k])

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            for x, target in zip(X, y):
                self.backpropagate(x, target, learning_rate)


# XOR Training Data
# Inputs: [0, 0], [0, 1], [1, 0], [1, 1]
# Outputs: [0], [1], [1], [0]
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# Create a network with 2 input neurons, 2 neurons in the hidden layer, and 1 output neuron
network = NeuralNetwork([
    Layer(2, 2),  # Hidden layer with 2 neurons, each taking 2 inputs
    Layer(1, 2)   # Output layer with 1 neuron, taking 2 inputs from the hidden layer
])

# Train the network
network.train(X, y, learning_rate=0.1, epochs=10000)

# Test the network
for input_data in X:
    print(
        f"Input: {input_data}, Predicted Output: {round(network.feedforward(input_data)[0])}")
