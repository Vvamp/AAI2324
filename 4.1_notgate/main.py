class NorPerceptron:
    def __init__(self, weights):
        self.weights = weights

    def activation(self, x):
        return int(not x)

    def output(self, inputs):
        cursum = 0
        for x in range(0, len(inputs)):
            cursum += inputs[x] * self.weights[x]
        return self.activation(cursum)


if __name__ == "__main__":
    nor_perceptron = NorPerceptron(weights=[-1, -1, -1])
    print("-- Nor Gate --")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            for x3 in [0, 1]:
                print(f"[{x1}, {x2}, {x3}]: {nor_perceptron.output([x1,x2,x3])}")
