from __future__ import annotations
from utils.math import sigmoid, sigmoid_derivative
import random
from typing import List


class Neuron:
    """
    Represents a neuron in a neural network.
    """

    def __init__(self, numInputs: int, neuronIndex: int):
        """Initializes a neuron

        Args:
            numInputs (int): The number of inputs for the neuron.
            neuronIndex (int): The index of the neuron in the layer.
        """
        self.weights = [random.uniform(0.0, 1.0) for _ in range(numInputs + 1)]
        self.bias = -1
        self.entrance = None
        self.delta = None
        self.neuronIndex = neuronIndex

    def input(self, inputs: List[float]):
        """
        Sets the inputs for the neuron.

        Args:
            inputs (List[float]): The input values for the neuron.
        """
        self.inputs = inputs
        self.inputs.append(self.bias)

    def output(self):
        """
        Computes the output of the neuron.

        Args:
            float: The output of the neuron.
        """
        if self.entrance == None:
            raise ValueError("entrance is not set")
        self.output_ = sigmoid(self.entrance)
        return self.output_

    def activate(self) -> float:
        """
        Calculates the input value (weighted sum) of the neuron and stores it inside the neuron(as entrance).
        """
        total = 0
        if len(self.inputs) != len(self.weights):
            raise ValueError(
                f"The number of inputs must match the number of weights, current len inputs is {len(self.inputs)} and len weights is {len(self.weights)}"
            )
        for inputIndex in range(len(self.inputs)):
            wi = self.weights[inputIndex]
            xi = self.inputs[inputIndex]
            total += wi * xi
        self.entrance = total

    def calculate_delta_other_layers(self, nextLayer: Layer):
        """
        Calculate the delta for the neuron in the hidden layer.

        Args:
            nextLayer (Layer): The next layer of the netwerk.

        """
        weightedDeltaSum = 0
        for nextNeuron in nextLayer.neurons:
            weightedDeltaSum += nextNeuron.weights[self.neuronIndex] * nextNeuron.delta
        self.delta = sigmoid_derivative(self.entrance) * weightedDeltaSum

    def calculate_delta_last_layer(self, expected: int):
        """
        Calculate the delta for the neuron in the output layer.

        Args:
            expected (int): The expected value of the neuron.

        """
        self.delta = sigmoid_derivative(self.entrance) * (expected - self.output_)

    def update_weights(self, learningRate: float):
        """
        Update the weights of the neuron by multiplying the learning rate with the delta and the input.

        Args:
            learningRate (float): The learning rate of the neuron.

        """
        for weightIndex in range(0, len(self.weights)):
            self.weights[weightIndex] += (
                learningRate * self.delta * self.inputs[weightIndex]
            )

    def update_bias(self, learningRate: float):
        """
        Update the bias of the neuron by multiplying the learning rate with the delta.

        Args:
            learningRate (float): The learning rate of the neuron.
        """
        self.bias += learningRate * self.delta
