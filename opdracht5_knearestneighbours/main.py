import numpy as np
import datetime
import matplotlib.pyplot as plt
import random
import typing

data = np.genfromtxt(
    "dataset1.csv",
    delimiter=";",
    usecols=[1, 2, 3, 4, 5, 6, 7],
    converters={
        5: lambda s: 0 if s == b"-1" else float(s),
        7: lambda s: 0 if s == b"-1" else float(s),
    },
)

datesasy = []
dates = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[0])
labels = []
for label in dates:
    # print(str(label)[4:5])
    ydate = datetime.datetime(
        int(str(label)[0:4]), int(str(label)[4:6]), int(str(label)[6:8])
    )
    ydate_start = datetime.datetime(2000, 1, 1)
    ydate_days = (ydate - ydate_start).days
    datesasy.append(ydate_days)

    if label < 20000301:
        labels.append("winter")
    elif 20000301 <= label < 20000601:
        labels.append("lente")
    elif 20000601 <= label < 20000901:
        labels.append("zomer")
    elif 20000901 <= label < 20001201:
        labels.append("herfst")
    else:  # from 01-12 to end of year
        labels.append("winter")


class FeatureVector:
    def __init__(self, feature_array: np.ndarray):
        self.FG = feature_array[0] / 10
        self.TG = feature_array[1] / 10
        self.TN = feature_array[2] / 10
        self.TX = feature_array[3] / 10
        self.SQ = feature_array[4] / 10
        self.DR = feature_array[5] / 10
        self.RH = feature_array[6] / 10


class TrainingSet:
    def __init__(self, vectors, labels):
        if len(vectors) != len(labels):
            raise ("Amount of labels doesn't match amount of vectors")

        self.data = []
        for i in range(0, len(vectors)):
            self.data.append((FeatureVector(vectors[i]), labels[i]))
        # print(data)


def computeDistance(
    vector_to_classify: FeatureVector, vector_to_compare: FeatureVector
):
    FG_d = abs(vector_to_classify.FG - vector_to_compare.FG)
    TG_d = abs(vector_to_classify.TG - vector_to_compare.TG)
    TN_d = abs(vector_to_classify.TN - vector_to_compare.TN)
    TX_d = abs(vector_to_classify.TX - vector_to_compare.TX)
    SQ_d = abs(vector_to_classify.SQ - vector_to_compare.SQ)
    DR_d = abs(vector_to_classify.DR - vector_to_compare.DR)
    RH_d = abs(vector_to_classify.RH - vector_to_compare.RH)

    return FG_d + TG_d + TN_d + TX_d + SQ_d + DR_d + RH_d


def classifier(training_set: TrainingSet, vector_to_classify: FeatureVector):
    instances_by_distance = []  # (Dist, Instance, Label)
    for xi, yi in training_set.data:
        instances_by_distance.append((computeDistance(vector_to_classify, xi), xi, yi))

    instances_by_distance.sort(key=lambda x: x[0])
    k = 5
    closest_instances = instances_by_distance[0:k]

    label_by_occurence = {}
    for close_instance in closest_instances:
        if close_instance[2] in label_by_occurence:
            label_by_occurence[close_instance[2]] += 1
        else:
            label_by_occurence[close_instance[2]] = 1

    # Return the most frequent class
    return max(label_by_occurence, key=label_by_occurence.get)


if __name__ == "__main__":
    # print(len(data))
    # print(len(labels))
    a = TrainingSet(data, labels)
    temps = []
    for day in a.data:
        # print("{} - {} degrees = {}".format(day[0].TN, day[0].TX, day[1]))
        temps.append(day[0].TG)

    b_data = np.array([85, 131, 120, 139, 37, 15, 20])  # data 19-11-2023
    b = FeatureVector(b_data)

    print(classifier(a, b))
    # Show input s
    # plt.scatter(datesasy, temps)
    # plt.ylabel("Avg Temp (C)")
    # plt.xlabel("Day of year")
    # plt.show()
