from __future__ import annotations
import random
from network.layer import Layer
from typing import List, NamedTuple
from utils.types import Dataset


class NeuralNetwork:
    """Represents a neural network with multiple, interconnected layers"""

    def __init__(self, learningRate: float = 0.01, seed: int = -1):
        """Create a neural network

        Args:
            learningRate (float, optional): The learning rate to use when training. Defaults to 0.01.
            seed (int, optional): The random seed to use. Defaults to -1 (Meaning no random seed is set).
        """
        self.layers: List[Layer] = []
        self.learningRate: float = learningRate
        if seed != -1:
            random.seed(42)

    def add_layer(self, layer: Layer):
        """Append a layer to the end of the neural network

        Args:
            layer (Layer): The layer to add to the neural network
        """
        self.layers.append(layer)

    def forward(self, inputs: List[float]):
        """Forward propagate the neural network

        Args:
            inputs (List[float]): Values for the input layer
        """
        currentInputs = inputs
        for layerIndex in range(0, len(self.layers)):
            currentInputs = self.layers[layerIndex].compute_results(currentInputs)

    def backward(self, expectedNeuronIndex: int):
        """Backpropagate the neural network

        Args:
            expectedNeuronIndex (int): The index of the output neuron that should have fired
        """
        self.layers.reverse()
        for layerIndex in range(0, len(self.layers)):
            layer = self.layers[layerIndex]
            if layerIndex == 0:
                layer.backpropagation_last_layer(expectedNeuronIndex)
            else:
                layer.backpropagation_other_layers(self.layers[layerIndex - 1])
        self.layers.reverse()

    def update_neuron_weights(self):
        """Update the neuron weights for each layer in the network"""
        for layer in self.layers:
            layer.update_neuron_weights(self.learningRate)

    def update_neuron_biases(self):
        """Update the neuron biases for each layer in the netwrok"""
        for layer in self.layers:
            layer.update_neuron_biases(self.learningRate)

    def train(
        self,
        trainingSet: Dataset,
        epochs: int,
    ):
        """Train the neural network on a given dataset

        Args:
            trainingSet (Dataset): The dataset to train on
            epochs (int): The number of epochs to train for
        """
        for epoch in range(0, epochs):
            for testIndex in range(0, len(trainingSet.inputs)):
                inputs = trainingSet.inputs[testIndex].tolist()
                expected = trainingSet.labels[testIndex]
                self.forward(inputs)
                self.backward(expected)
                self.update_neuron_weights()
            self.update_neuron_biases()

    def test(self, testDataset: Dataset) -> float:
        """Test the neural network on a given dataset

        Args:
            testDataset (Dataset): A dataset to test the neural network on

        Returns:
            float: The correctness percentage(rounded to 2 decimals) of the neural network
        """
        correct = 0
        total = len(testDataset.labels)
        for testIndex in range(0, len(testDataset.inputs)):
            inputs = testDataset.inputs[testIndex].tolist()
            expected = testDataset.labels[testIndex]
            self.forward(inputs)
            completeNeurons = self.layers[-1].neurons
            maxNeuronValue = max(completeNeurons, key=lambda neuron: neuron.output())
            for neuronIndex in range(0, len(completeNeurons)):
                if completeNeurons[neuronIndex] == maxNeuronValue:
                    if neuronIndex == expected:
                        correct += 1
                    break
        return round(correct / total * 100, 2)

    @staticmethod
    def create(
        neuronsPerHiddenLayer: List[int],
        inputNeurons: int,
        outputNeurons: int,
        learningRate: float = 0.01,
        seed: int = -1,
    ) -> NeuralNetwork:
        """Create a neural network with the given parameters

        Args:
            neuronsPerHiddenLayer (List[int]): A list of neuron counts, meaning the amount of neurons per hidden layer
            inputNeurons (int): How many input neurons the network should have
            outputNeurons (int): How many output neurons the network should have
            learningRate (float, optional): What learning rate the network should use when training. Defaults to 0.01.
            seed (int, optional): What random seed to use, if any. Defaults to -1 (meaning no random seed).

        Raises:
            ValueError: If the learning rate is negative or 0

        Returns:
            NeuralNetwork: An instantiated neural network
        """
        network = NeuralNetwork(learningRate, seed)
        for hiddenLayerIndex in range(len(neuronsPerHiddenLayer)):
            if neuronsPerHiddenLayer[hiddenLayerIndex] <= 0:
                raise ValueError("hiddenLayer must be positive and above 0")
            if len(network.layers) == 0:
                network.add_layer(
                    Layer(neuronsPerHiddenLayer[hiddenLayerIndex], inputNeurons)
                )
            else:
                network.add_layer(
                    Layer(
                        neuronsPerHiddenLayer[hiddenLayerIndex],
                        neuronsPerHiddenLayer[hiddenLayerIndex - 1],
                    )
                )
        network.add_layer(Layer(outputNeurons, neuronsPerHiddenLayer[-1]))
        return network

    @staticmethod
    def find_best_learning_rate(
        epochs: int,
        maxLearningRate: float,
        trainData: Dataset,
        testData: Dataset,
        seed=-1,
    ) -> float:
        """Find the best learning rate for a predefined neural network

        Args:
            epochs (int): How many epochs should be used in training
            maxLearningRate (float): What the highest learning rate is that will be tested
            trainData (Dataset): A training dataset
            testData (Dataset): A test dataset
            seed (int, optional): The random seed to use. Defaults to -1 (meaning no random seed is set).

        Raises:
            ValueError: When the maxLearningRate is negative or 0
            ValueError: When epochs is negative or 0

        Returns:
            float: The best learning rate found
        """
        if maxLearningRate <= 0:
            raise ValueError("maxLearningRate must be positive and above 0")
        if epochs <= 0:
            raise ValueError("epochs must be positive and above 0")

        maxLearningRate = maxLearningRate * 1000  # Make it small
        maxCorrectness, maxCorrectLearningRate = 0, 0

        # Try out all the learning rates
        for learningRate in range(1, int(maxLearningRate), 1):
            learningRate = learningRate / 1000

            # Create the neural network
            network = NeuralNetwork.create(
                [6, 3], trainData.inputs.shape[1], 3, learningRate, seed
            )
            # Train the network
            network.train(trainData, epochs)

            # Check how correct the network is
            correctness = network.test(testData)
            if correctness > maxCorrectness:
                maxCorrectness = correctness
                maxCorrectLearningRate = learningRate
            if correctness == 100:
                break

        return maxCorrectLearningRate
