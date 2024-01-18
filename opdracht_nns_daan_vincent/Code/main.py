from __future__ import annotations
from sklearn.datasets import load_iris
from network.network import NeuralNetwork
from utils.data import SplitData
if __name__ == "__main__":
    # How many epochs to train
    epochsToTrain = 69
    randomSeed = 42
    learningRate = 0.159
    # Split the data into training and test sets
    trainingSet, testSet = SplitData(datasetFunction=load_iris, randomSeed=randomSeed)
    
    # Find a good learning rate (uncomment to find the best learning rate)
    # learningRate = NeuralNetwork.find_best_learning_rate(
    #     epochs=epochsToTrain,
    #     maxLearningRate=0.5,
    #     trainData=trainingSet,
    #     testData=testSet,
    # )

    # Test how correct our network is with the given learningRate
    network = NeuralNetwork.create(
        neuronsPerHiddenLayer=[3,6],
        inputNeurons=trainingSet.inputs.shape[1],
        outputNeurons=3,
        learningRate=learningRate,
        seed=randomSeed,
    )
    network.train(trainingSet, epochsToTrain)

    correctness = network.test(testSet)
    print(
        f"Correctness: {correctness}% with learning rate of {learningRate}",
    )
