from sklearn import svm
from .feature import Feature


class SVM(Feature):
    """
    DESCRIPTITON:
        scikit-learn Support vector machine wrapper

    PARAMS:
        :gamma: kernel coefficient
    """

    def __init__(self, gamma=.001):
        Feature.__init__(self, 'Support Vector Machine')
        self.classifier = svm.SVC(gamma=gamma)

    def train(self, X, y):
        """
        DESCRIPTION:
            train the svm

        PARAMS:
            :X: the data
            :y: the labels for the data
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """
        DESCRIPTION:
            predrict values for new data

        PARAMS:
            :X: data to predict labels for
        """
        return self.classifier.predict(X)
