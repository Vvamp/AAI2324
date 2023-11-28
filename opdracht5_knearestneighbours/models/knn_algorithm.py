from ast import Tuple
from typing import List, Tuple
from data import weather_dataset
from models import classifier
from models.feature_vector import FeatureVector

class KnnAlgorithm:
    """This class combines the functions and data that are required for the K-Nearest Neighbors Algorithm
    """

    def __init__(self, training_dataset: weather_dataset.Dataset, processing_dataset: weather_dataset.Dataset):
        """Initializes a K-Nearest Neighbors algorithm based on a training/classification dataset and a dataset that needs processing

        Args:
            training_dataset (Dataset): The dataset to train on
            processing_dataset (Dataset): The dataset to process
        """
        self.training_dataset = training_dataset
        self.processing_dataset = processing_dataset
        # Normalize all datasets that need to be normalized(based on the value passed to their constructor)
        weather_dataset.Dataset.normalize([self.training_dataset, self.processing_dataset])

    def find_best_kvalue(self, max_iterations: int = 250) -> int:
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
            current_iteration_success_percentage = self.get_success_percentage(
                current_iteration_k_value)
            current_iteration_k_value += 1
            k_results.append(current_iteration_success_percentage)

        # The highest success percentage is deemed the best k value
        best_k_percentage = max(k_results)
        best_k_value = k_results.index(best_k_percentage) + 1
        return best_k_value

    def get_success_percentage(self, k_value: int = 58) -> float:
        """Classifies each element in the processing dataset, based on the training dataset using the passed k_value, and checks if they are correct

        Args:
            k_value (int, optional): The amount of nearest neighbors to use for classification. Defaults to 58.

        Returns:
            float: The percentage(from 0.0 to 100.0) of correct entries out of the total in the processing dataset
        """
        if self.processing_dataset.hasLabels == False:
            raise ValueError("Passed processing dataset has no labels. Labels are needed to check for success percentage.")
        total_entries = len(self.processing_dataset.vectors)
        correct_entries = 0.0

        for element_index in range(0, len(self.processing_dataset.vectors)):
            testdata = self.processing_dataset.vectors[element_index]
            testdata_label = self.processing_dataset.labels[element_index]
            knn_classifier = classifier.Classifier(self.training_dataset, testdata, k_value)
            classifier_result = knn_classifier.get_vector_classification()
            if classifier_result == testdata_label:
                correct_entries += 1.0

        return correct_entries / total_entries * 100
    
    def get_classifications_for_data(self, k_value: int = 58) -> List[Tuple[FeatureVector, str]]:
        classifications : List[Tuple[FeatureVector, str]] = []
        for element_index in range(0, len(self.processing_dataset.vectors)):
            vector_to_classify = self.processing_dataset.vectors[element_index]
            knn_classifier = classifier.Classifier(self.training_dataset, vector_to_classify, k_value)
            classifier_result = knn_classifier.get_vector_classification()
            classifications.append((vector_to_classify, classifier_result))
        return classifications

