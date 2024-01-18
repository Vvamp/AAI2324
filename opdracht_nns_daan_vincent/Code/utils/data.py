from sklearn.model_selection import train_test_split
from utils.types import Dataset


def SplitData(datasetData = None, datasetTarget = None, datasetFunction = None, testSize = 0.2, randomSeed = -1):
    """ Split the data into training and test sets
    
    Args:
        datasetData (np.ndarray): The data of the dataset if datasetFunction is None
        datasetTarget (np.ndarray): The target of the dataset if datasetFunction is None
        datasetFunction (function): The function to get the dataset if datasetData and datasetTarget are None
        testSize (float): The size of the test set
        randomSeed (int): The random seed to use 
    
    Returns:
        Dataset: The training set
        Dataset: The test set
    
    """
    if datasetFunction is not None:
        if datasetData is not None or datasetTarget is not None:
            raise Exception("datasetData and datasetTarget should be None if datasetFunction is set")
        dataset = datasetFunction()
        data = dataset.data
        target = dataset.target
    elif datasetData is not None and datasetTarget is not None:
        data = datasetData
        target = datasetTarget
    else:
        raise Exception("Either datasetData and datasetTarget should be set or datasetFunction should be set")
    xTrain, xTest, yTrain, yTest = train_test_split(
        data, target, test_size=testSize, random_state=randomSeed
    )
    trainingSet = Dataset(xTrain, yTrain)
    testSet = Dataset(xTest, yTest)
    return trainingSet, testSet