from typing import List


class NandPerceptron:
    def __init__(self, weights, bias=-1):
        self.weights = weights
        self.weights.append(bias)

    def activation(self, cursum):
        return int(not cursum < 0)

    def output(self, inputs: List[float]):
        inputs.append(-1)
        cursum = 0
        for i in range(0, len(inputs)):
            cursum += inputs[i] * self.weights[i]
        return self.activation(cursum)


def adder(x1: int, x2: int):
    nand = NandPerceptron(weights=[-1, -1])
    l1 = nand.output([x1, x2])
    l2a = nand.output([x1, l1])
    l2b = nand.output([x2, l1])
    out_sum = nand.output([l2a, l2b])
    out_carry = nand.output([l1, l1])

    print(f"{x1} + {x2} = ({out_carry}){out_sum} ")


if __name__ == "__main__":
    nand_perceptron = NandPerceptron(weights=[0.5, 0.5])
    print("-- Nand Gate --")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"[{x1}, {x2}]: {nand_perceptron.output([x1,x2])}")
    print("Adder:")
    adder(0, 0)
    adder(1, 0)
    adder(0, 1)
    adder(1, 1)
