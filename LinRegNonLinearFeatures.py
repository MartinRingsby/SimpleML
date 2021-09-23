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

class NumpyLinRegClass(NumpyClassifier):

    def fit(self, X_train, t_train, gamma = 0.1, epochs=500, diff = 0.00001):
        #Ser at accuracyen ikke går høyere enn 0.6075 uansett hvor mye vi reduserer diff
        #etter 0.00001 så da ser det ut til at denne er den optimale verdien
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""

        x1_squared = np.zeros((len(X_train),1))
        x2_squared = np.zeros((len(X_train),1))
        x1_x2 = np.zeros((len(X_train),1))

        for i in range(len(X_train)):
            x1_squared[i] = X_train[i][0]**2
            x2_squared[i] = X_train[i][1]**2
            x1_x2[i] = X_train[i][0]*X_train[i][1]



        X_train = np.concatenate([X_train, x1_squared], axis = 1)
        X_train = add_bias(X_train)
        print(X_train)

        (k, m) = X_train.shape

        self.theta = np.zeros(m)
        prevTheta = 0

        for e in range(epochs):
            thetaSum = 0
            self.theta -= gamma / k *  X_train.T @ (X_train @ self.theta - t_train)
            for i in range(len(self.theta)):
                thetaSum += self.theta[i]
            if (abs(prevTheta - thetaSum) <= diff):
                break
            prevTheta = thetaSum
        print(self.theta)
        print('epochs: ' , e)

    def predict(self, x, threshold=0.5):
        z = x
        z1_squared = np.zeros((len(z),1))
        z2_squared = np.zeros((len(z),1))
        z1_z2 = np.zeros((len(z),1))

        for i in range(len(z)):
            z1_squared[i] = z[i][0]**2
            z2_squared[i] = z[i][1]**2
            z1_z2[i] = z[i][0]*z[i][1]


        print(self.theta)
        z = np.concatenate([z, z1_squared], axis = 1)
        z = add_bias(z)


        score = z @ self.theta

        return score>threshold



lin_cl = NumpyLinRegClass()
lin_cl.fit(X_train, t2_train)

a = lin_cl.accuracy(X_val, t2_val)
print(a)
