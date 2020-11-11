import numpy as np
import struct
from helpers import load_mnist_images, load_mnist_labels 
from svm_linear import SVM_linear
import pandas as pd
import argparse

def remove_zeros(ser, breast):
    if breast:
        ser['diagnosis'] = 1 if ser['diagnosis'] == 1 else -1
    else:
        ser['y'] = 1 if ser['y'] == 1 else -1
    return ser

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--kernel', help='linear or rbf', required=True)
parser.add_argument('--dataset', help='MNIST or  bcd', required=True)
parser.add_argument('--train', help='path to training data', required=True)
parser.add_argument('--test', help='path to test data', required=True)
parser.add_argument('--output', help='path to output file', required=False)
args = vars(parser.parse_args())
print(args, "args")

kernel = args['kernel']
dataset = args['dataset']
train = args['train']
test = args['test']
output = args['output']

def linear_sim(x1, x2):
    return np.dot(x1, x2)

def dist(x1, x2):
    return np.sum(x1-x2)

def rbf_sim(x1, x2):
    return np.exp(-0.7 * (dist(x1,x2) **2))



# data['x0'] = np.ones([data.shape[0], 1])
# data = data[['x0', *(list(data.columns.values)[:-1])]]
# print(data.head())
class SVM_rbf():
    def __init__(self, X, Y):
        # self.data = data
        self.X_data = X
        self.Y_data = Y
        self.weight = None


    def train(self,T, lambdaa):
        C = 0.25
        # alphas = np.zeros(self.X_data.shape[0])
        betas = np.zeros(self.X_data.shape[0])
        Ws = []
        for t in range(1, T):
            alphas = ( betas / (lambdaa * t))  
            Ws.append(alphas)
            i = np.random.randint(0, self.X_data.shape[0])
            score = 0
            for j in range(self.X_data.shape[0]):
                if(i != j):
                    # vote -> weight * label * similarity 
                    score += alphas[j] * rbf_sim(self.X_data[i], self.X_data[j]) 
            # print(score, "Scroe", t)
            if score * self.Y_data[i]  < 1:
                betas[i] += self.Y_data[i] 
           
            self.weight = (1/T) * np.array(Ws).sum(axis=0)
        print(self.weight)


            
        return self.weight

    def predict(self,X):
        
        Y_pred = np.ones(X.shape[0])

        for i in range(X.shape[0]):
            score = 0
            for j in range(self.X_data.shape[0]):
                
                # vote -> weight * label * similarity 
                score += self.weight[j] * self.Y_data[j] * rbf_sim(X[i], self.X_data[j]) 
            if score <=0:
                Y_pred[i] = -1
        return Y_pred

    def accuracy(self, X, Y):
        
        Y_pred = self.predict(X)
        
        return (Y[Y!=Y_pred].shape[0] / Y.shape[0])

if dataset == 'bcd': # Breast cancer
    if kernel == "rbf":
        data = pd.read_csv(train + "\\Breast_cancer_data.csv")
        model = SVM_rbf(data.values[:, :-1], data.values[:, -1])
        model.train(2000, 1)
        print(model.accuracy(data.values[:, :-1], data.values[:, -1]))
    else:
        data = pd.read_csv(train +"\\Breast_cancer_data.csv")
        print(data.shape, "shape")
        model = SVM_linear()
        model.fit(data.values[:, :-1], data.values[:, -1], 2000)
        y_pred = model.predict(data.values[:, :-1], data.values[:, -1])

def MNIST_model(model_i):
    model_i = 2
    X_train = load_mnist_images( train , "train").reshape(-1, 28*28)
    print(X_train.shape, "SSS")
    X_test = load_mnist_images(test, "test").reshape(-1, 28*28)
    Y_train = load_mnist_labels(train,"train")
    Y_test = load_mnist_labels(test,"test")
    Y_train[Y_train!=model_i] = -1
    Y_train[Y_train == model_i] = 1
    Y_test[Y_test!=model_i] = -1
    Y_test[Y_test == model_i] = 1

    if kernel == 'rbf':
        model = SVM_rbf(X_train, Y_train)
        model.train(4000, 1)
        print(model.accuracy(X_test, Y_test))
    else:
        model = SVM_linear()
        model.fit(X_train, Y_train, 2000)
        y_pred = model.predict(X_test, Y_test)

if dataset == 'MNIST':
    for i in range(10):
        MNIST_model(i)       












# import matplotlib.pyplot as plt
# images = load_mnist_images("train")
# plt.imshow(images[0,:,:], cmap='gray')
# plt.show()

# labels = load_mnist_labels("train")

# print(labels)
