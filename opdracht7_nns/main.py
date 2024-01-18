from typing import List
import math


def sigmoid(val):
    return 1 / (1 + pow(math.e, -val))


def cost(outputs: List[float], expected_values: List[float]):
    if len(outputs) != len(expected_values):
        raise Exception("Output and Expected Values length are not equal!")
    error_sum = 0
    for i in range(0, len(outputs)):
        error_sum += (expected_values[i] - outputs[i]) ** 2
    MSE = error_sum / len(outputs)
    return MSE


class Neuron:
    def __init__(self, weights, bias=-1):
        self.weights = weights
        self.weights.append(bias)
        self.inputs = []

    def activation(self, cursum):
        return sigmoid(cursum)

    def input(self, inputs: List[float]):
        self.inputs = inputs

    def output(self):
        current_inputs = self.inputs + [-1]
        current_sum = 0
        for i in range(0, len(current_inputs)):
            current_sum += current_inputs[i] * self.weights[i]
        return self.activation(current_sum)

    def update(self, inputs: List[float]):
        pass


if __name__ == "__main__":
    n = Neuron([0.5, 0.5, 0.5])
    n.input([0, 0, 0])
    print(n.output())
