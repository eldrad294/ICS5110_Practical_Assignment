#
# Library Imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#
class LogisticRegressor:
    #
    def __init__(self, features_matrix, labels):
        self.features = features_matrix
        self.labels = labels
        self.classifier = LogisticRegression()
    #
    def fit_training_set(self):
        """ Fits the training dataset to the Logistic Regression classifier """
        self.classifier.fit(self.features, self.labels)
    #
    def predict(self, feature_array):
        """ Predicts the passed values according to the established Logistic Regression classifier """
        return self.classifier.predict(feature_array)
    #
    def calculate_accuracy_score(self, labels, predicted):
        """" Calculates the accuracy score based on the predicted """
        return accuracy_score(labels, predicted)