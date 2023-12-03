from typing import List
import math


def sigmoid(val):
    return 1 / (1 + pow(math.e, -val))


class Neuron:
    def __init__(self, weights, bias=-1, learning_rate=0.1):
        self.weights = weights
        self.weights.append(bias)
        self.learning_rate = learning_rate

    def activation(self, cursum):
        return sigmoid(cursum)

    def output(self, inputs: List[float]):
        curinputs = inputs + [-1]
        cursum = 0
        for i in range(0, len(curinputs)):
            cursum += curinputs[i] * self.weights[i]
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


if __name__ == "__main__":
    print("-- Nor Gate --")
    inputs = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
    results = [
        1, 0, 0, 0, 0
    ]

    n = Neuron([0.5, 0.5, 0.5])
    passes = 1
    while True:
        isCorrect = True
        for i in range(0, len(inputs)):
            n.update(inputs[i], results[i])
            print(
                f"- {inputs[i]} = {round(n.output(inputs[i]))} (should be {results[i]})")
            if round(n.output(inputs[i])) != results[i]:
                isCorrect = False
        print("")

        if isCorrect:
            print(f"Done! Passes: {passes}")
            break
        passes += 1
    print("-- AND Gate --")
    inputs = [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1]
    ]
    results = [
        0, 0, 0, 0, 1, 0, 0,
    ]

    n = Neuron([0.5, 0.5, 0.5])
    passes = 1
    correctRows = 0
    while True:
        isCorrect = True
        for i in range(0, len(inputs)):
            n.update(inputs[i], results[i])
            print(
                f"- {inputs[i]} = {round(n.output(inputs[i]))} (should be {results[i]})")
            if round(n.output(inputs[i])) != results[i]:
                isCorrect = False
        print("")

        if isCorrect and correctRows >= 10:
            print(f"Done! Passes: {passes}")
            break
        passes += 1
        if not isCorrect:
            correctRows = 0
        else:
            correctRows += 1
        if passes > 1000:
            print("No consistent correctness after 1000 passes")
            break
    print(round(n.output([0, 0, 0])), " should be (0)")

    print(round(n.output([0, 0, 1])), " should be (0)")

    print(round(n.output([0, 1, 0])), " should be (0)")

    print(round(n.output([0, 1, 1])), " should be (0)")

    print(round(n.output([1, 0, 0])), " should be (0)")

    print(round(n.output([1, 0, 1])), " should be (0)")

    print(round(n.output([1, 1, 0])), " should be (0)")
    print(round(n.output([1, 1, 1])), " should be (1)")

    print(n.weights)
