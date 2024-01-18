from __future__ import annotations
from utils.math import sigmoid, sigmoid_derivative
from network.neuron import Neuron
from typing import List
import copy


class Layer:
    def __init__(self, numNeurons, numInputsPerNeuron):
        """
        Represents a layer of neurons in a neural network.

        Args:
            numNeurons (int): The number of neurons in the layer.
            numInputsPerNeuron (int): The number of inputs per neuron.
        """
        self.neurons = [
            Neuron(numInputsPerNeuron, neuronIndex)
            for neuronIndex in range(numNeurons)
        ]

    def get_deltas(self) -> List[float]:
        """Get the delta's for each neuron in the layer.

        Returns:
            List[float]: A list of delta's for each neuron in the layer.s
        """
        deltas = []
        for neuron in self.neurons:
            deltas.append(neuron.delta)
        return deltas

    def compute_results(self, inputs: List[float]):
        """
        Computes the results of the layer based on the given inputs.

        Args:
            inputs (List[float]): The input values for the layer.

        Returns:
            List[float]: The computed results of the layer.
        """
        results = []
        for neuron in self.neurons:
            neuron.input(inputs[:])
            neuron.activate()
            results.append(neuron.output())
        return results

    def backpropagation_last_layer(self, expectedNeuronIndex: int):
        """Backpropagate the output layer

        Args:
            expectedNeuronIndex (int): The index of the output neuron that should have fired
        """
        for neuronIndex in range(0, len(self.neurons)):
            self.neurons[neuronIndex].calculate_delta_last_layer(
                expectedNeuronIndex == neuronIndex
            )

    def backpropagation_other_layers(self, nextLayer: Layer):
        """Backpropagate a non-output layer

        Args:
            nextLayer (Layer): The next layer in the network
        """
        for neuronIndex in range(0, len(self.neurons)):
            self.neurons[neuronIndex].calculate_delta_other_layers(nextLayer)

    def update_neuron_weights(self, learningRate: float):
        """Update the neuron weights for each neuron in the layer

        Args:
            learningRate (float): The learning rate to use to update the weights
        """
        for neuron in self.neurons:
            neuron.update_weights(learningRate)

    def update_neuron_biases(self, learningRate: float):
        """Update the neuron biases for each neuron in the layer

        Args:
            learningRate (float): The learning rate to use to update the biases
        """
        for neuron in self.neurons:
            neuron.update_bias(learningRate)
