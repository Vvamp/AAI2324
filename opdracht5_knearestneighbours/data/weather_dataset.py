from __future__ import annotations
from models import feature_vector
import numpy as np
from typing import Tuple, List
import math

class Dataset:
    """This class specifies a weather data dataset
    """

    def __init__(self, filename: str, normalizeInput: bool, containsLabels: bool, year: int):
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
        self.vectors : List[feature_vector.FeatureVector] = []
        # Labels are the season for each vector, where labels[i] should correspond to vectors[i]
        self.labels : List[str] = []
        self.raw_array_data : np.ndarray = np.array([])  # Raw numpy array data
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
            self.vectors.append(feature_vector.FeatureVector(data_row))

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
        min_val_per_column = [math.inf for column_index in range(
            0, len(training_sets[0].raw_array_data[0]))]
        max_val_per_column = [-math.inf for column_index in range(
            0, len(training_sets[0].raw_array_data[0]))]

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
        min_val_per_column, max_val_per_column = Dataset.calculate_min_max(
            training_sets)
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
                training_set.vectors.append(feature_vector.FeatureVector(data_row))
