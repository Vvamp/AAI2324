
from ast import Tuple
from data import weather_dataset
from models import feature_vector 
from typing import List, Tuple
import math 

class Classifier:
    """This class combines the functions to classify a feature vector, based on a given dataset as training data
    """

    def __init__(self, training_set: weather_dataset.Dataset, vector_to_classify: feature_vector.FeatureVector, k_value: int = 58):
        """Initializes a classifier object. Needs a training dataset and an unclassified feature vector

        Args:
            training_set (Dataset): The dataset to train on
            vector_to_classify (FeatureVector): The feature vector for which to generate a classification for
            k_value (int, optional): The amount of closest neighbors from the training set to compare the feature vector to. Defaults to 58.
        """
        self.training_set = training_set
        self.vector_to_classify = vector_to_classify
        self.k_value = k_value

    def compute_distance_euclidean(self, training_vector: feature_vector.FeatureVector) -> float:
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
            instances_by_distance = [
            ]  # List of tuple with elements: (Distance to , vector compared against, )
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
        def get_k_closest_instances(instances_by_distance: List[Tuple[float, feature_vector.FeatureVector, str]]):
            """Returns the 'k' amount of closest instances of training vectors based on their distance

            Args:
                instances_by_distance (List[(float, FeatureVector, str)]): A list of tuples, where each tuple has the computed distance between the training vector and the unclassified feature vector, the training feature vector itself and the training feature vector label

            Returns:
                List[(float, FeatureVector, str)]: A list of tuples, where each tuple has the computed distance between the training vector and the unclassified feature vector, the training feature vector itself and the training feature vector label
            """
            instances_by_distance.sort(key=lambda instance: instance[0])
            return instances_by_distance[0:self.k_value]

        # Count occurrences by label
        def get_label_occurrence_counts(closest_instances: List[Tuple[float, feature_vector.FeatureVector, str]]):
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

   