#
# Library Imports
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#
class Naive_Bayes:
    #
    def __init__(self, features_matrix, labels):
        self.features = features_matrix
        self.labels = labels
        self.classifier = GaussianNB()
    #
    def fit_training_set(self):
        """ Fits the training dataset to the Naive Bayes classifier """
        self.classifier.fit(self.features, self.labels)
    #
    def predict(self, feature_array):
        """ Predicts the passed values according to the established Naive Bayes classifier """
        return self.classifier.predict(feature_array)
    #
    def calculate_accuracy_score(self, labels, predicted):
        """" Calculates the accuracy score based on the predicted """
        return accuracy_score(labels, predicted)