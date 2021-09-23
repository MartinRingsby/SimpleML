import numpy as np
import sklearn
from sklearn import datasets
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from collections import Counter

X, t = make_blobs(n_samples=[400,800,400], centers=[[0,0],[1,2],[2,3]],
n_features=2, random_state=2019)

indices = np.arange(X.shape[0])
random.seed(2020)
random.shuffle(indices)
indices[:10]

X_train = X[indices[:800],:]
X_val = X[indices[800:1200],:]
X_test = X[indices[1200:],:]

t_train = t[indices[:800]]
t_val = t[indices[800:1200]]
t_test = t[indices[1200:]]

t2_train = t_train == 1
t2_train = t2_train.astype('int')
t2_val = (t_val == 1).astype('int')
t2_test = (t_test == 1).astype('int')


def add_bias(X):
    # Put bias in position 0
    sh = X.shape
    if len(sh) == 1:
        #X is a vector
        return np.concatenate([np.array([1]), X])
    else:
        # X is a matrix
        m = sh[0]
        bias = np.ones((m,1)) # Makes a m*1 matrix of 1-s
        return np.concatenate([bias, X], axis  = 1)


class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""

    def accuracy(self,X_test, y_test, **kwargs):
        pred = self.predict(X_test, **kwargs)
        if len(pred.shape) > 1:
            pred = pred[:,0]

        #print(y_test)
        #print(pred)
        return sum(pred==y_test)/len(pred)




def logistic(x):
    return 1/(1+np.exp(-x))

class NumpyLogReg(NumpyClassifier):


    def fit(self, X_train, t_train, gamma = 0.1, diff=0.000001):
        #The accuracy of the classifier stops at the exact same point as the
        #linear regressor but it takes way longer for the logistic regressor
        #to get to this accuracy the diff value of 0.0000001 get the best
        #accuracy
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""
        self.classes = np.unique(t_train)

        X_train = add_bias(X_train)
        (k, m) = X_train.shape

        """We want a list of the arrays were 1 signifies that the item belongs to the class, and zero if
        it belongs to any of the other classes"""
        self.t_classes = []
        """Check each number in the array to see if the number corresponds to the class"""
        for i in range(len(self.classes)):
            true_or_false = t_train == self.classes[i]
            true_or_false_ints = true_or_false.astype('int')
            self.t_classes.append(true_or_false_ints)



        self.theta = np.zeros((len(self.classes), m))
        for i in range(len(self.classes)):
            norm = 1
            while norm>diff:
                self.theta[i,:] -= gamma / k *  X_train.T @ (self.forward(X_train, i) - self.t_classes[i])
                change = gamma / k *  X_train.T @ (self.forward(X_train, i) - self.t_classes[i])
                norm = np.linalg.norm(change)



    def forward(self, X, i):
        return logistic(X @ self.theta[i])

    def predict(self, x):
        z = add_bias(x)
        score = np.zeros((len(z), len(self.theta[0])))

        for i in range(len(z)):
            for j in range(len(self.theta[0])):
                score[i,j] = self.forward(z[i], j)

        predicted = np.zeros(len(score))

        for i in range(len(score)):
            predicted[i] = np.argmax(score[i])

        return predicted


    def confusion(self, t_test, x_test):
        predicted = self.predict(x_test)
        precision = np.zeros(len(self.theta[0]))
        recall = np.zeros(len(self.theta[0]))
        confusion_matrix = np.zeros((len(self.theta[0]), len(self.theta[0])))


        for i in range(len(predicted)):
            confusion_matrix[int(predicted[i]), int(t_test[i])] += 1


        for i in range(len(precision)):
            precision[i] = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
            recall[i] = confusion_matrix[i][i]/np.sum(confusion_matrix.T[i])

        return confusion_matrix, recall, precision









log_cl = NumpyLogReg()
log_cl.fit(X_train, t_train)
AccuracyLogReg = log_cl.accuracy(X_test, t_test)
Confusion, recall, precision = log_cl.confusion(t_test, X_test)
print(Confusion, recall, precision)

print('The Accuracy of the Logistical Regression Classifier: ', AccuracyLogReg)
