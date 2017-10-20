from sklearn import svm
from sklearn.metrics import accuracy_score
#
class SVM:
    #
    def __init__(self, features_matrix, labels, kernel, gamma=None, C=1):
        self.features = features_matrix
        self.labels = labels
        if gamma is None:
            self.classifier = svm.SVC(kernel=kernel, C=C)
        else:
            self.classifier = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    #
    def fit_training_set(self):
        """ Fits the training dataset to the SVM classifier """
        self.classifier.fit(self.features, self.labels)
    #
    def predict(self, feature_array):
        """ Predicts the passed values according to the established SVM classifier """
        return self.classifier.predict(feature_array)
    #
    def calculate_accuracy_score(self, labels, predicted):
        """" Calculates the accuracy score based on the predicted """
        return accuracy_score(labels, predicted)