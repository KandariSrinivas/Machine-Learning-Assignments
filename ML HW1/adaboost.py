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

flag = True if df.values.shape[1] > 5 else False
df = df.apply(lambda ser: remove_zeros(ser, flag), axis=1)

class WeakClass:
    def __init__(self):
        self.parity = 1
        self.threshold = None
        self.col = None
        self.alpha = None

    def predict(self, X):
        preds = np.ones([X.shape[0]])
        
        if(self.parity == 1):
            preds[X[:, self.col] < self.threshold] = -1
        else:
            preds[X[:, self.col] > self.threshold] = -1  
        return preds
        

## Define X and Y
class Adaboost:
    def __init__(self, T):
        self.T = T
        self.models = []
    def fit(self, X, Y):


        W = np.ones(X.shape[0]) * (1/X.shape[0])
        # models = []
        for i in range(self.T):
            model = WeakClass()
            minError = np.inf
            for col in range(X.shape[1]):
                for threshold in np.unique(X[:, col]):
                    model.threshold = threshold
                    model.col = col
                    preds = model.predict(X)
                    error = np.sum(W[preds != Y])
                    ## if error is more than 0.5 then take complement 1- error
                    if error > 0.5:
                        model.parity = -1
                        error = 1- error
                    ## record best col, threshold
                    if error < minError:
                        minError = error
                        bestParity = model.parity
                        bestCol = model.col
                        bestThres = model.threshold
            model.parity = bestParity
            model.threshold = bestThres
            model.col = bestCol
            model.alpha = 0.5 * np.log((1.0 - minError + (1e-10)) / (minError + (1e-10)))
            preds = model.predict(X)
            self.models.append(model)
            W = W * np.exp(-1 * model.alpha * Y * preds)
            W = W / np.sum(W)

    def predict(self, X):
        preds = np.array([model.alpha * model.predict(X) for model in self.models])
        y_preds = np.sum(preds, axis=0)
        y_preds = np.sign(y_preds)
        return y_preds


if(mode == "erm"):
    X = df.values[:, :-1]
    Y = df.values[:, -1]

    model = Adaboost(6)
    model.fit(X, Y)

    k=0
    for m in model.models:
        print( "Weak learner", k,"'s alpha for ERM", m.alpha)
        k += 1

    y_preds = model.predict(X)
    print("Error ERM: ", y_preds[y_preds != Y].shape[0]/ y_preds.shape[0])



### K- Fold
if(mode == "cross_valid"):
    print("############################################ K-FOLD ###################################")
    data = df.values
    bins = args["nfold"]
    # print(data, "data")
    bin_size = int(data.shape[0]/bins)
    np.random.shuffle(data)
    errors = []
    for i in range(bins):
        val_set = data[i*bin_size: (i+1) * bin_size , :]
        train = np.delete(data, list(range(i*bin_size, (i+1) * bin_size)), axis=0)
        model = Adaboost(6)
        model.fit(train[:,:-1], train[:,-1])
        y = val_set[:,-1]
        y_pred = model.predict(val_set[:,:-1])
        # print("Model ", i , "s weights", "Alpha: ", model.alpha, "Column: ", model.col, "Threshold: ", model.threshold)
        k=0
        for m in model.models:
            print( "Weak learner", k,"'s alpha for ", i, "th fold", m.alpha)
            k += 1
        err = y[y * y_pred < 0].shape[0] / y.shape[0]
        print("Error on ", i, "th Fold", err)
        errors.append(err)
    print("Average Error on 10 Folds", np.mean(np.array(errors)))
