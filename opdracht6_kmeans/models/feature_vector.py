import numpy as np

class FeatureVector:
    """This class specifies feature vector definition used for the weather datasets
    """

    def __init__(self, feature_array: np.ndarray):
        """Initializes the feature vector based on a np.ndarray with 7 elements. Each element should match the specification on the Canvas assignment 5 page.

        Args:
            feature_array (np.ndarray): The array with weather data for a single day
        """
        self.FG = feature_array[0]
        self.TG = feature_array[1]
        self.TN = feature_array[2]
        self.TX = feature_array[3]
        self.SQ = feature_array[4]
        self.DR = feature_array[5]
        self.RH = feature_array[6]

    def __len__(self) -> int:
        """The length of a feature vector should be equal to the values it contains, in this case 7

        Returns:
            int: The length of the feature vector
        """
        return 7

