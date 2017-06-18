from sklearn import svm
from .feature import Feature


class SVM(Feature):

    def __init__(self, gamma=.001):
        Feature.__init__(self, 'Support Vector Machine')
        self.classifier = svm.SVC(gamma=gamma)

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
