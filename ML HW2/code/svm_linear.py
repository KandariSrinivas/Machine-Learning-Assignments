import numpy as np
class SVM_linear:
    def __init__(self, alpha=0.001, lambdaa=0.01):
        self.alpha = alpha
        self.lambdaa = lambdaa
        
    def fit(self, X, Y, T):
        print("fitting")
        feats = X.shape[1]
        self.w = np.zeros(feats)
        self.b = 0
        for t in range(T):
            i = np.random.randint(0, X.shape[0])
                
            score = Y[i] * (np.dot(X[i], self.w) - self.b)
            if score >=1 :
                self.w -= self.alpha * (2 * self.lambdaa * self.w)
            else:
                self.w -= self.alpha * (2 * self.lambdaa * self.w - (X[i] * Y[i]))
                self.b -= self.alpha * Y[i]
    def predict(self, X, y):
        
        count = X.shape[0]
        Y_pred = np.ones(count)
        for i in range(count):
            score = np.dot(X[i], self.w) - self.b
            Y_pred[i] = np.sign(score)
        print("Error: ", (Y_pred[y!= Y_pred].shape[0]) / count)
        return Y_pred