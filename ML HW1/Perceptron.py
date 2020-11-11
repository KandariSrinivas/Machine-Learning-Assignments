import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import argparse


def remove_zeros(ser, breast):
    if breast:
        ser['diagnosis'] = 1 if ser['diagnosis'] == 1 else -1
    else:
        ser['y'] = 1 if ser['y'] == 1 else -1
    return ser
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='CSV dataset file location', required=True)
parser.add_argument('--mode', help='erm or cross_valid', required=True)
parser.add_argument('--nfold', help='erm or cross_valid', default=10, required=False)
args = vars(parser.parse_args())

# comment this for breast cancer 
df = pd.read_csv(args['dataset'])
mode =  args["mode"]

df['x0'] = np.ones([df.shape[0], 1])
df = df[['x0', *(list(df.columns.values)[:-1])]]

flag = True if df.values.shape[1] > 5 else False
df = df.apply(lambda ser: remove_zeros(ser, flag), axis=1)


class Perceptron():
    def __init__(self, data):
        self.W = np.zeros([data.shape[1]-1])
        self.data = data

    def predict(self, X):
        return X.dot(self.W)

    def train(self, epochs):
        
        
        for e in range(epochs):      
            for i in range(len(self.data)):
                X = self.data[i, :-1]
                Y = self.data[i, -1]

                Y_pred = X.dot(self.W)
                
                if(Y_pred * Y <= 0):
                    self.W = self.W + X * Y
                  
            Y_pred = self.data[:,:-1].dot(self.W)
            Y = self.data[:,-1]
            total=0
            count=0 
            for a, b in zip(Y, Y_pred):
                if(a * b < 0):
                    count += 1
                total += 1
            # print(count/total, count,  "err")
            if(count==0):
                return self.W
            # print("Epoch: ", e, self.W)

if(mode == "erm"):
    model = Perceptron(df.values)
    model.train(50)
    X = df.values[:, :-1]
    Y = df.values[:, -1]
    Y_pred = model.predict(X)
    error = Y_pred[Y*Y_pred < 0].shape[0]/ Y_pred.shape[0]
    print("ERM weights", model.W)
    print("Error ERM ", error)
# Breast cancer ERM weights [ 4.810000e+02  3.715118e+03 -1.192400e+03  1.910020e+04 -2.895000e+03 1.220189e+01]
# linearly sep ERM weights [-10.         -40.02891027  20.0112814 ]
### K- Fold
print("############################################ K-FOLD ###################################")
if(mode == "cross_valid"):
    data = df.values
    bins = args["nfold"]
    # print(data, "data")
    bin_size = int(data.shape[0]/bins)
    np.random.shuffle(data)
    errors = []
    for i in range(bins):
        val_set = data[i*bin_size: (i+1) * bin_size , :]
        train = np.delete(data, list(range(i*bin_size, (i+1) * bin_size)), axis=0)
        model = Perceptron(train)
        model.train(50)
        y = val_set[:,-1]
        y_pred = model.predict(val_set[:,:-1])
        print("Model ", i , "th Fold's weights", model.W)
        err = y[y * y_pred < 0].shape[0] / y.shape[0]
        print("Error on ", i, "th Fold", err)
        errors.append(err)

    print("Average Error on 10 Folds", np.mean(np.array(errors)))