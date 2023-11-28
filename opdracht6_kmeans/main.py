from data import weather_dataset
from models import knn_algorithm


if __name__ == "__main__":
    training_set = weather_dataset.Dataset("dataset1.csv", False, True, 2000) 
    validation_set = weather_dataset.Dataset("validation1.csv", False, True, 2001)
    dataset_to_classify = weather_dataset.Dataset("days.csv", False, True, 2001)
    
    # Find best K value and calculate success rate for that value with the validation.csv
    validation_algorithm = knn_algorithm.KnnAlgorithm(training_set, validation_set)
    print("Calculating implementation's success rate and the best K value")
    best_k_value = validation_algorithm.find_best_kvalue()
    print("Success Rate: {}%".format(validation_algorithm.get_success_percentage(best_k_value)))
    
    # Classify Days.csv
    classification_algorithm = knn_algorithm.KnnAlgorithm(training_set, dataset_to_classify)
    classifications = classification_algorithm.get_classifications_for_data(best_k_value)
    for classification in classifications:
        print("{} classified as {}".format(classification[0], classification[1]))
