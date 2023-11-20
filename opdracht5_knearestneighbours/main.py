import numpy as np
import datetime
import matplotlib.pyplot as plt
import random
import typing

from math import sqrt
import math


class FeatureVector:
    def __init__(self, feature_array: np.ndarray):
        self.FG = feature_array[0]
        self.TG = feature_array[1]
        self.TN = feature_array[2]
        self.TX = feature_array[3]
        self.SQ = feature_array[4]
        self.DR = feature_array[5]
        self.RH = feature_array[6]


class TrainingSet:
    def __init__(self, vectors, labels):
        if len(vectors) != len(labels):
            raise ("Amount of labels doesn't match amount of vectors")

        self.data = []
        for i in range(0, len(vectors)):
            self.data.append((FeatureVector(vectors[i]), labels[i]))
        # print(data)


def calculate_min_max(training_sets: list[np.ndarray]):
    min_val_per_column = [math.inf for col in range(0, len(data[0]))]
    max_val_per_column = [-math.inf for col in range(0, len(data[0]))]
    for training_set in training_sets:
        for column_index in range(0, 7):
            min_val = min(data[:, column_index])
            if min_val < min_val_per_column[column_index]:
                min_val_per_column[column_index] = min_val

            max_val = max(data[:, column_index])
            if max_val > max_val_per_column[column_index]:
                max_val_per_column[column_index] = max_val
    return (min_val_per_column, max_val_per_column)


def normalize(data: np.ndarray, min_val_per_column, max_val_per_column):
    new_columns = []
    for column_index in range(0, 7):
        min_val = min_val_per_column[column_index]
        max_val = max_val_per_column[column_index]
        new_column = []
        for cell in data[:, column_index]:
            newvalue = (cell - min_val) / (max_val - min_val) * 100
            new_column.append(newvalue)
            # print(
            #     "Normalized value from {} to {} for min {}/max {} ".format(
            #         cell, newvalue, min_val, max_val
            #     )
            # )
        new_columns.append(new_column)

    normalized_data_inverted = np.array(new_columns)
    normalized_data = normalized_data_inverted.transpose()
    return normalized_data


def loadDataSet(filename: str, year: int = 2000):
    shifted_year_value = year * 10000
    data = np.genfromtxt(
        filename,
        delimiter=";",
        usecols=[1, 2, 3, 4, 5, 6, 7],
        converters={
            5: lambda s: 0 if s == b"-1" else float(s),
            7: lambda s: 0 if s == b"-1" else float(s),
        },
    )
    # data = normalize(data)

    dates = np.genfromtxt(filename, delimiter=";", usecols=[0])
    labels = []
    for label in dates:
        # print("CMP {} to {}".format(label, shifted_year_value + 1201))
        if label < shifted_year_value + 301:
            labels.append("winter")
        elif shifted_year_value + 301 <= label < shifted_year_value + 601:
            labels.append("lente")
        elif shifted_year_value + 601 <= label < shifted_year_value + 901:
            labels.append("zomer")
        elif shifted_year_value + 901 <= label < shifted_year_value + 1201:
            labels.append("herfst")
        else:  # from 01-12 to end of year
            labels.append("winter")
    return (data, labels)


def computeDistance(
    vector_to_classify: FeatureVector, vector_to_compare: FeatureVector
):
    # # Diff between all and add the diffs
    # FG_d = abs(vector_to_classify.FG - vector_to_compare.FG)
    # TG_d = abs(vector_to_classify.TG - vector_to_compare.TG)
    # TN_d = abs(vector_to_classify.TN - vector_to_compare.TN)
    # TX_d = abs(vector_to_classify.TX - vector_to_compare.TX)
    # SQ_d = abs(vector_to_classify.SQ - vector_to_compare.SQ)
    # DR_d = abs(vector_to_classify.DR - vector_to_compare.DR)
    # RH_d = abs(vector_to_classify.RH - vector_to_compare.RH)
    # return FG_d + TG_d + TN_d + TX_d + SQ_d + DR_d + RH_d

    d2 = (
        (vector_to_classify.FG - vector_to_compare.FG) ** 2
        + (vector_to_classify.TG - vector_to_compare.TG) ** 2
        + (vector_to_classify.TN - vector_to_compare.TN) ** 2
        + (vector_to_classify.TX - vector_to_compare.TX) ** 2
        + (vector_to_classify.SQ - vector_to_compare.SQ) ** 2
        + (vector_to_classify.DR - vector_to_compare.DR) ** 2
        + (vector_to_classify.RH - vector_to_compare.RH) ** 2
    )
    return math.sqrt(d2)  # This is squared bu that might be fine


def classifier(training_set: TrainingSet, vector_to_classify: FeatureVector, k_value=5):
    instances_by_distance = []  # (Dist, Instance, Label)

    # Calculate distance to each vector in the training set from the vector to classify
    for training_vector, training_label in training_set.data:
        instances_by_distance.append(
            (
                computeDistance(vector_to_classify, training_vector),
                training_vector,
                training_label,
            )
        )

    # Sort by instances by distance and get the first 'k' closest
    instances_by_distance.sort(key=lambda x: x[0])
    closest_instances = instances_by_distance[0:k_value]

    # Count occurrences by label
    label_by_occurrence = {}
    for close_instance in closest_instances:
        if close_instance[2] in label_by_occurrence:
            label_by_occurrence[close_instance[2]] += 1
        else:
            label_by_occurrence[close_instance[2]] = 1

    # Return the most frequent class
    return max(label_by_occurrence, key=label_by_occurrence.get)


def GetSuccessPercentage(dataset: TrainingSet, testdataset: TrainingSet, k_value=5):
    total = len(testdataset.data)
    correct = 0.0
    error = 0.0
    for testdata, testdata_label in testdataset.data:
        classifier_result = classifier(dataset, testdata, k_value)
        if classifier_result == testdata_label:
            correct += 1.0
            # print("Result {} is correct".format(classifier_result))
        else:
            # print(
            #     "Result {} is incorrect. Should be {}".format(
            #         classifier_result, testdata_label
            #     )
            # )
            error += 1.0

    return correct / total * 100


if __name__ == "__main__":
    (data, labels) = loadDataSet("dataset1.csv", 2000)
    (validation_data, validation_labels) = loadDataSet("validation1.csv", 2001)

    # Normalize
    (min_val_per_column, max_val_per_column) = calculate_min_max(
        [data, validation_data]
    )
    data = normalize(data, min_val_per_column, max_val_per_column)
    validation_data = normalize(validation_data, min_val_per_column, max_val_per_column)

    dataset = TrainingSet(data, labels)
    # testdata = np.array([85, 131, 120, 139, 37, 15, 20])  # data 19-11-2023
    testdataset = TrainingSet(validation_data, validation_labels)

    print(GetSuccessPercentage(dataset, testdataset, 5))

    res = 1
    current_k = 1
    k_results = []
    print("Finding best K value")
    while res < 100 and res > 0 and current_k < 500:
        res = GetSuccessPercentage(dataset, testdataset, current_k)
        print("K: {} = {}%".format(current_k, res))
        current_k += 1
        k_results.append(res)
    # testvector = FeatureVector(testdata)
    best_k = max(k_results)
    best_k_index = k_results.index(best_k) + 1

    print("Best result", max(k_results), "% for k={}".format(best_k_index))
    # print(classifier(dataset, testvector))
