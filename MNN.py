import numpy as np
import sklearn
from sklearn import datasets
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from collections import Counter
from sklearn.preprocessing import MinMaxScaler



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

scaler = MinMaxScaler()
scaler.fit(X_train)
scaler.fit(X_val)
scaler.fit(X_test)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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


class MNNClassifier():
    """A multi-layer neural network with one hidden layer"""

    def __init__(self, eta = 0.001, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.eta = eta
        self.dim_hidden = dim_hidden

        # Should you put additional code here?

    def fit(self, X_train, t_train, epochs = 100):
        """Intialize the weights. Train *epochs* many epochs."""
        self.X_train = add_bias(X_train)
        self.dim_in = len(self.X_train[0])
        self.dim_out = np.max(t_train)+1
        self.y_train = t_train

        self.weights1 = np.zeros((self.dim_in, self.dim_hidden))
        for i in range(self.dim_in):
            for e in range(self.dim_hidden):
                self.weights1[i,e] = random.uniform(-1,1)

        self.weights2 = np.zeros((self.dim_hidden+1, self.dim_out))
        for i in range(self.dim_hidden+1):
            for e in range(self.dim_out):
                self.weights2[i, e] = random.uniform(-1,1)
        # Initialization
        # Fill in code for initialization

        print('Weights1 before epochs: ', self.weights1,'Weights1 before epochs: ', self.weights2)

        for e in range(epochs):
            # Run one epoch of forward-backward
            #Fill in the code
            """Shuffling the list each epoch"""
            X_and_Y = np.zeros((len(self.X_train), len(self.X_train[0])+1))
            for i in range(len(self.X_train)):
                X_and_Y[i,-1] = self.y_train[i]
                for j in range(len(self.X_train[i])):
                    X_and_Y[i,j] = self.X_train[i,j]

            np.random.shuffle(X_and_Y)

            X_shuffled = np.zeros((len(self.X_train), len(self.X_train[0])))
            Y_shuffled = np.zeros(len(self.y_train))

            for i in range(len(self.X_train)):
                for j in range(len(self.X_train[i])):
                    X_shuffled[i,j] = X_and_Y[i,j]
            for i in range(len(self.y_train)):
                Y_shuffled[i] = X_and_Y[i,-1]


            X_random_order = np.random.shuffle(self.X_train)
            for i in range(len(self.X_train)):
                self.backwards(X_shuffled[i], Y_shuffled[i])
            pass

        print('Weights1 after epochs: ', self.weights1,'Weights1 after epochs: ', self.weights2)

    def forward(self, X):
        """Perform one forward step.
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""

        hidden_activations_before = np.zeros(len(self.weights1[0]))
        hidden_activations_before = X @ self.weights1
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

    def backwards(self, X_train_vector, y_train_vector):
        outputs, hidden = self.forward(X_train_vector)
        X_train = add_bias(X_train_vector)

        y_vector = np.zeros(np.max(self.y_train)+1)
        y_vector[int(y_train_vector)] = 1
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



    def accuracy(self, X_test, t_test):
        """Calculate the accuracy of the classifier on the pair (X_test, t_test)
        Return the accuracy"""
        OutActivations = np.zeros((len(X_test), self.dim_out))
        X_test = add_bias(X_test)
        predictions = np.zeros(len(t_test))

        score = 0
        for i in range(len(X_test)):
            OutActivations[i], HiddenActivations = self.forward(X_test[i])
            index_max = np.argmax(OutActivations[i])
            print(OutActivations[i])
            print(index_max)

            predictions[i] = index_max
            if t_test[i] == predictions[i]:
                score += 1

        accuracy = score/len(t_test)

        print(score)
        print(accuracy)

        return accuracy







Neural = MNNClassifier()
Neural.fit(X_train, t_train)
Neural.accuracy(X_val, t_val)
