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
        return np.concatenate([np.array([-1]), X])
    else:
        # X is a matrix
        m = sh[0]
        bias = np.ones((m,1)) # Makes a m*1 matrix of 1-s
        biasNegative = np.negative(bias)
        return np.concatenate([biasNegative, X], axis  = 1)




def mse(y, y_pred):
    sum_errors = 0.
    for i in range(0,len(y)):
        sum_errors += (y[i] - y_pred[i])**2
    mean_squared_error = sum_errors/len(y)
    return mean_squared_error

def errorOut(k, y, t):
    error = np.zeros(k)
    for i in range(k):
        error[i] = (y[i]-t[i])*y[i]*(1-y[i])
    return error

def errorHidden(k, hiddenAct, weights, error):
    hiddenError = np.zeros(k)
    for i in range(k):
        sumWeightsError = 0
        for j in range(len(error)):
            sumWeightsError += weights[i,j]*error[j]
        hiddenError[i] = hiddenAct[i]*(1-hiddenAct[i])*sumWeightsError
    return hiddenError



def sigmoid(x):
    return (1/(1+np.exp(-x)))

class MLP():
    def __init__(self, X_train, Y_train, dim_hidden=6, eta=0.01):
        self.dim_hidden = dim_hidden
        self.eta = eta
        self.X_train = add_bias(X_train)
        self.dim_in = len(self.X_train[0])
        self.P = len(self.X_train)

        dim_out = np.max(Y_train)+1
        
        self.y_train = Y_train
        self.weights1 = np.zeros((self.dim_in, self.dim_hidden))

        for i in range(self.dim_in):
            for e in range(self.dim_hidden):
                self.weights1[i,e] = random.uniform(-1,1)

        self.weights2 = np.zeros((self.dim_hidden+1, dim_out))
        for i in range(self.dim_hidden+1):
            for e in range(dim_out):
                self.weights2[i, e] = random.uniform(-1,1)

    def forward(self, X_train):

        X_train = add_bias(X_train)
        hidden_activations_before = np.zeros(len(self.weights1[0]))
        hidden_activations_before = X_train @ self.weights1
        hidden_activations_after = np.zeros(len(hidden_activations_before))


        for k in range(len(hidden_activations_after)):
            hidden_activations_after[k] = sigmoid(hidden_activations_before[k])


        hidden_activations_after = add_bias(hidden_activations_after)

        """Take the activation values from the hidden layer, multiply them with the
        weights2 according to each synapse and find the activation value of the
        output layer before running it through the sigmoid"""

        out_activations_before = np.zeros(len(self.weights2[0]))
        out_activations_before = hidden_activations_after @ self.weights2
        out_activations_after = np.zeros(len(out_activations_before))


        """Run each value through the sigmoid and we get our activation values"""
        for k in range(len(out_activations_after)):
            out_activations_after[k] = sigmoid(out_activations_before[k])

        return out_activations_after, hidden_activations_after

    def backwards(self, X_train, y_train):
        outputs, hidden = self.forward(X_train)
        X_train = add_bias(X_train)

        y_vector = np.zeros(np.max(self.y_train)+1)
        y_vector[y_train] = 1
        print(y_vector)
        lossOutput = errorOut(len(outputs), outputs, y_vector)
        lossHidden = errorHidden(len(hidden), hidden, self.weights2, lossOutput)

        old_weights2 = self.weights2.copy()


        for i in range(len(self.weights2)):
            for count, element in enumerate(self.weights2[i]):
                change = self.eta*lossOutput[count]*hidden[i]
                self.weights2[i, count] = element - change
        new_weights2 = self.weights2



        old_weights1 = self.weights1.copy()
        for i in range(len(self.weights1)):
            for count, element in enumerate(self.weights1[i]):
                change = self.eta*lossHidden[count]*X_train[i]
                self.weights1[i, count] = element - change

        new_weights1 = self.weights1

        return old_weights2, new_weights2, old_weights1, new_weights1, lossOutput, lossHidden


MLP = MLP(X_train, t_train)
old2, new2, old1, new1, LossOut, LossHidden = MLP.backwards(X_train[0],t_train[0])
print('The old weights from the hidden layer to the outputlayer was: ')
print(old2)
print('The new weights from the hidden layer to the outputlayer is now: ')
print(new2)
print('The old weights from the inputlayer to the hidden layer was:')
print(old1)
print('The new weights from the inputlayer to the hidden layer was:')
print(new1)
