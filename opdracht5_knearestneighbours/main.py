from __future__ import annotations
import numpy as np
import typing
import math


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


class Dataset:
    """This class specifies a weather data dataset
    """
    def __init__(self, filename : str, normalizeInput : bool, containsLabels : bool, year : int):
        """Initializes the dataset object based on a csv file

        Args:
            filename (str): The path to the dataset file in csv format
            normalizeInput (bool): Whether this datasets input should be normalized
            containsLabels (bool): Whether this dataset has 'labels' for the elements, which are dates in the first column (used to specify the season)
            year (int): The year this dataset is from
        """
        self.filename = filename
        self.doNormalize = normalizeInput
        self.hasLabels = containsLabels
        self.year = year
        self.vectors = None
        self.labels = None # Labels are the season for each vector, where labels[i] should correspond to vectors[i]
        self.raw_array_data = None # Raw numpy array data
        self.load_dataset()
        self.load_dataset_labels()

    def load_dataset(self):
        """Load the feature vectors from the dataset file
        """
        self.raw_array_data = np.genfromtxt(
            self.filename,
            delimiter=";",
            usecols=[1, 2, 3, 4, 5, 6, 7],
            converters={
                5: lambda s: 0 if s == b"-1" else float(s),
                7: lambda s: 0 if s == b"-1" else float(s),
            },
        )
        
        # Load the raw numpy data into 'FeatureVector' objects for readability
        self.vectors = [] 
        for data_row in self.raw_array_data:
            self.vectors.append(FeatureVector(data_row))

    def load_dataset_labels(self):
        """Generate labels for the dataset based on the date columns
        """
        # If the csv has labels, load those in as well
        if self.hasLabels:
            dates = np.genfromtxt(self.filename, delimiter=";", usecols=[0])
            self.labels = []
            for label in dates:
                if label < int(str(self.year) + "0301"):
                    self.labels.append("winter")
                elif int(str(self.year) + "0301") <= label < int(str(self.year) + "0601"):
                    self.labels.append("lente")
                elif int(str(self.year) + "0601") <= label < int(str(self.year) + "0901"):
                    self.labels.append("zomer")
                elif int(str(self.year) + "0901") <= label < int(str(self.year) + "1201"):
                    self.labels.append("herfst")
                else:  # from 01-12 to end of year
                    self.labels.append("winter")

    @staticmethod
    def calculate_min_max(training_sets: list[Dataset]) -> Tuple[List[float], List[float]]:
        """Calculates the minimum and maximum values for each column in the datasets

        Args:
            training_sets (list[Dataset]): All datasets find the minimum and maximum values for

        Returns:
            Tuple[List[float], List[float]]: A tuple with 2 lists: a list with the minimum and maximum values for each column, respectively
        """
        # Initialize an array with a max/min value for each column in the dataset
        min_val_per_column = [math.inf for column_index in range(0, len(training_sets[0].raw_array_data[0]))]
        max_val_per_column = [-math.inf for column_index in range(0, len(training_sets[0].raw_array_data[0]))]

        # For each column in each training set, overwrite the min/max if they are lower or higher respectively
        for training_set in training_sets:
            for column_index in range(0, len(training_set.raw_array_data[0])):
                min_val = min(training_set.raw_array_data[:, column_index])
                if min_val < min_val_per_column[column_index]:
                    min_val_per_column[column_index] = min_val

                max_val = max(training_set.raw_array_data[:, column_index])
                if max_val > max_val_per_column[column_index]:
                    max_val_per_column[column_index] = max_val

        # Return the lowest/highest value for each column in the dataset found over all training sets
        return (min_val_per_column, max_val_per_column)

    @staticmethod
    def normalize(training_sets: list[Dataset]):
        """Normalize each dataset that has the 'doNormalize' flag set in their constructor, based on the minimum and maximum values for each column in all datasets

        Args:
            training_sets (list[Dataset]): All datasets to include in the minimum and maximum value calculations. Only datasets with the 'doNormalize' flag set are actually affected.
        """
        new_columns = []
        min_val_per_column, max_val_per_column = Dataset.calculate_min_max(training_sets)
        for training_set in training_sets:
            if training_set.doNormalize == False:
                continue 

            for column_index in range(0, len(training_sets[0].raw_array_data[0])):
                min_val = min_val_per_column[column_index]
                max_val = max_val_per_column[column_index]
                new_column = []
                for cell in training_set.raw_array_data[:, column_index]:
                    newvalue = (cell - min_val) / (max_val - min_val) * 100
                    new_column.append(newvalue)
                new_columns.append(new_column)

            normalized_data_inverted = np.array(new_columns)
            normalized_data = normalized_data_inverted.transpose()
            training_set.vectors.clear()
            for data_row in normalized_data:
                training_set.vectors.append(FeatureVector(data_row))

class Classifier:
    """This class combines the functions to classify a feature vector, based on a given dataset as training data
    """
    def __init__(self, training_set : Dataset, vector_to_classify : FeatureVector, k_value = 58 : int):
        """Initializes a classifier object. Needs a training dataset and an unclassified feature vector

        Args:
            training_set (Dataset): The dataset to train on
            vector_to_classify (FeatureVector): The feature vector for which to generate a classification for
            k_value (int, optional): The amount of closest neighbors from the training set to compare the feature vector to. Defaults to 58.
        """
        self.training_set = training_set
        self.vector_to_classify = vector_to_classify
        self.k_value = k_value

    def compute_distance_euclidean(self, training_vector: FeatureVector) -> float:
        """Calculate the distance between a single vector from the training data and the vector that needs to be classified

        Args:
            training_vector (FeatureVector): A single (already-classified) vector from the training data. 

        Returns:
            float: The Euclidean distance between the training vector and the vector that needs to be classified 
        """
        distance_squared = (
            (self.vector_to_classify.FG - training_vector.FG) ** 2
            + (self.vector_to_classify.TG - training_vector.TG) ** 2
            + (self.vector_to_classify.TN - training_vector.TN) ** 2
            + (self.vector_to_classify.TX - training_vector.TX) ** 2
            + (self.vector_to_classify.SQ - training_vector.SQ) ** 2
            + (self.vector_to_classify.DR - training_vector.DR) ** 2
            + (self.vector_to_classify.RH - training_vector.RH) ** 2
        )
        return math.sqrt(distance_squared)


    def get_vector_classification(self) -> float:
        """Computes the classification for the feature vector passed in the constructor, based on the training dataset also passed in the constructor
        """
        def get_instances_by_distance():
            """Computes the distance between the feature vector and training feature vector, for each vector within the training dataset, and returns these in a tuple along with the training vector itself and the label

            Returns:
                List<(float, FeatureVector, str)>: A list of tuples, where each tuple has the computed distance between the training vector and the unclassified feature vector, the training feature vector itself and the training feature vector label
            """
            instances_by_distance = []  # List of tuple with elements: (Distance to , vector compared against, )
            # Calculate distance to each vector in the training set from the vector to classify
            for element_index in range(0, len(self.training_set.vectors)):
                training_vector = self.training_set.vectors[element_index]
                training_label = self.training_set.labels[element_index]
                instances_by_distance.append(
                    (
                        self.compute_distance_euclidean(training_vector),
                        training_vector,
                        training_label,
                    )
                )
            return instances_by_distance

        # Sort by instances by distance and get the first 'k' closest
        def get_k_closest_instances(instances_by_distance : List[(float, FeatureVector, str)]):
            """Returns the 'k' amount of closest instances of training vectors based on their distance

            Args:
                instances_by_distance (List[(float, FeatureVector, str)]): A list of tuples, where each tuple has the computed distance between the training vector and the unclassified feature vector, the training feature vector itself and the training feature vector label

            Returns:
                List[(float, FeatureVector, str)]: A list of tuples, where each tuple has the computed distance between the training vector and the unclassified feature vector, the training feature vector itself and the training feature vector label
            """
            instances_by_distance.sort(key=lambda instance: instance[0])
            return instances_by_distance[0:self.k_value]

        # Count occurrences by label
        def get_label_occurrence_counts(closest_instances : List[(float, FeatureVector, str)]):
            """Counts the amount of times each label occurs in the given list

            Args:
                closest_instances (List[(float, FeatureVector, str)]): A list of tuples, where each tuple has the computed distance between the training vector and the unclassified feature vector, the training feature vector itself and the training feature vector label

            Returns:
                Dictionary(str, int): A dictionary where the key is the label and the value is the amount the label has occurred in the list
            """
            label_by_occurrence = {}
            for close_instance in closest_instances:
                if close_instance[2] in label_by_occurrence:
                    label_by_occurrence[close_instance[2]] += 1
                else:
                    label_by_occurrence[close_instance[2]] = 1
            return label_by_occurrence

        instances_by_distance = get_instances_by_distance()
        closest_instances = get_k_closest_instances(instances_by_distance)
        label_by_occurrence = get_label_occurrence_counts(closest_instances)
        # Return the most frequent class
        return max(label_by_occurrence, key=label_by_occurrence.get)


class KnnAlgorithm:
    """This class combines the functions and data that are required for the K-Nearest Neighbors Algorithm
    """
    def __init__(self, training_dataset : Dataset, processing_dataset : Dataset):
        """Initializes a K-Nearest Neighbors algorithm based on a training/classification dataset and a dataset that needs processing

        Args:
            training_dataset (Dataset): The dataset to train on
            processing_dataset (Dataset): The dataset to process
        """
        self.training_dataset = training_dataset
        self.processing_dataset = processing_dataset
        Dataset.normalize([self.training_dataset, self.processing_dataset]) # Normalize all datasets that need to be normalized(based on the value passed to their constructor)
       
        
    def find_best_kvalue(self, max_iterations=250 : int) -> int:
        """Attempts to find the best k_value for the processing dataset(can only be done if the 'processing dataset' has labels, as these are used to verify the results)z

        Args:
            max_iterations (int, optional): _description_. Defaults to 250.

        Returns:
            int: The k_value found
        """
        current_iteration_success_percentage = 1
        current_iteration_k_value = 1
        k_results = []
        
        # Run the 'get_success_percentage' for each value of K between 1 and 'max_iterations' (step_size = 1) and append the result in 'k_results'
        while current_iteration_k_value < max_iterations:
            current_iteration_success_percentage = self.get_success_percentage(current_iteration_k_value)
            current_iteration_k_value += 1
            k_results.append(current_iteration_success_percentage)

        # The highest success percentage is deemed the best k value
        best_k_percentage = max(k_results)
        best_k_value = k_results.index(best_k_percentage) + 1
        return best_k_value
        
    def get_success_percentage(self, k_value=58 : int) -> float:
        """Classifies each element in the processing dataset, based on the training dataset using the passed k_value, and checks if they are correct

        Args:
            k_value (int, optional): The amount of nearest neighbors to use for classification. Defaults to 58.

        Returns:
            float: The percentage(from 0.0 to 100.0) of correct entries out of the total in the processing dataset
        """
        total_entries = len(self.processing_dataset.vectors)
        correct_entries = 0.0

        for element_index in range(0, len(self.processing_dataset.vectors)):
            testdata = self.processing_dataset.vectors[element_index]
            testdata_label = self.processing_dataset.labels[element_index]
            classifier = Classifier(self.training_dataset, testdata, k_value)
            classifier_result = classifier.get_vector_classification()
            if classifier_result == testdata_label:
                correct_entries += 1.0

        return correct_entries / total_entries * 100

if __name__ == "__main__":
    training_set = Dataset("dataset1.csv", False, True, 2000)
    validation_set = Dataset("validation1.csv", False, True, 2001)
    algo = KnnAlgorithm(training_set, validation_set)
    algo.find_best_kvalue()
    print("Success Rate: {}%".format(algo.get_success_percentage()))