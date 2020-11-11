import argparse
import pandas as pd
import numpy  as np
import random
import math
import csv
from sklearn.cluster import KMeans

def transformTrainingDataPerDigit(trainingData, labels, reqDigit):
    dataRelatedToDigit = []
    for x,y in zip(trainingData, labels):
        if y == reqDigit:
            dataRelatedToDigit.append(x)

    trainingDataForDigit = np.array(dataRelatedToDigit)
    return trainingDataForDigit

class GMM:
    def __init__(self, X, K = 3):
        self.k = int(K)
        #For each of the components, we have component weight c_k, mean vector m_k, and
        #covariance matrix cv_k
        self.componentWeights = [] # List of components weights of all k components
        self.componentCenters = [] # List of components centers of all k components
        self.componentCovarianceMatrices = [] #List of components covariances of all k components

        numObservations, numFeatures = X.shape

        for _ in range(self.k):
            self.componentWeights.append(1.0/self.k)

        #np.conv gives normalized covariance, we need unnormalized. So, multiplying accordingly
        covariance = np.cov(X) * (numObservations - 1)

        for _ in range(self.k):
            covarianceMatrix = np.zeros((numFeatures,numFeatures), dtype=float)
            for i in range(numFeatures):
                covarianceMatrix[i][i] = covariance[i][i]
            self.componentCovarianceMatrices.append(covarianceMatrix)


        #These are already pre calculated using K-means ++ algorithm.
        #These are the cluster centers obtained after running K-means++ .
        #There is good research literature available justifying that, initializing our initial centers before running EM
        #algorithm with those values lead to better GMM results
        if (self.k == 1):
            self.componentCenters[0] = np.mean(X, axis=0)
        else:
            kmeans = KMeans(n_clusters = self.k, init ='k-means++').fit(X)
            for j in range(self.k):
                self.componentCenters.append(kmeans.cluster_centers_[j,:])

    def update(self, componentsWeights, componentsCenters, componentsCovarianceMatrices):
        self.componentWeights = componentsWeights
        self.componentCenters = componentsCenters
        self.componentCovarianceMatrices = componentsCovarianceMatrices

class GMMForAllDigits:
    def __init__(self, K = 3):
        self.components = int(K)
        self.GMMForDigits = None
    

    def fit(self, trainingData,Y):
        self.GMMForDigits = []
        maxEpochs = 1000

        for digit in range(10):
            X = transformTrainingDataPerDigit(trainingData, Y ,digit)
            numObservations, numFeatures = X.shape

            #Obtain a initial GMM model
            GMMForSingleDigit = GMM(X, self.components)
            componentsWeights = GMMForSingleDigit.componentWeights
            componentsCenters = GMMForSingleDigit.componentCenters
            componentsCovarianceMatrices = GMMForSingleDigit.componentCovarianceMatrices

            #Run EM algorithm
            r = np.zeros((self.components, numObservations), dtype=float)
            for _ in range(maxEpochs):
                for i in range(self.components):
                    for j in range(numObservations):
                        covMatrixInv = np.linalg.pinv(GMMForSingleDigit.componentCovarianceMatrices[i])
                        det          = np.linalg.det(GMMForSingleDigit.componentCovarianceMatrices[i]) + 0.0000001
                        xMinusCenter = np.subtract(X[j,:],GMMForSingleDigit.componentCenters[i])
                        xMinusCenter = xMinusCenter.reshape(1,64)
                        te = np.dot(xMinusCenter, np.dot(covMatrixInv, np.transpose(xMinusCenter)))

                        r[i][j] = componentsWeights[i] * ((1/(2*math.pi)) ** (numFeatures/2)) * (1.0/math.sqrt(det)) *\
                                  math.exp(-0.5 * te[0][0])

                    totalSum = np.sum(r, axis =1)
                    r[i,:] = r[i,:]/totalSum[i]
                
                #Updating centers, component weights and components Covariance matrices
                for i in range(self.components):
                    sumTemp = 0
                    sumr = 0
                    covTemp = np.zeros((numFeatures,numFeatures), dtype=float)
                    for j in range(numObservations):
                        sumTemp += r[i][j]*X[j,:]
                        sumr += r[i][j]
                        covTemp = np.add(((np.dot( np.transpose(X[j,:] - componentsCenters[i]), X[j,:] - componentsCenters[i])) * r[i][j]), covTemp)
                    componentsCenters[i] = sumTemp/sumr
                    componentsWeights[i] = sumr/numObservations
                    componentsCovarianceMatrices[i] = covTemp/sumr

            GMMForSingleDigit.update(componentsWeights, componentsCenters, componentsCovarianceMatrices)
            self.GMMForDigits.append(GMMForSingleDigit)

    def predict(self,X,Y):
        correctlyPredicted = 0
        numObservations = X.shape[0]
        correctlyPredictedPerDigit = [0] * 10
        countPerDigit = [0] * 10

        for (x,y) in zip(X,Y):
            predictedDigit = 0
            actualDigit = int(y)
            maxProbabilityMeasure = 0
            for digit in range(10):
                probabilityMeasure = 0
                GMMForSingleDigit = self.GMMForDigits[digit]
                for component in range(GMMForSingleDigit.k):
                    det = np.linalg.det(GMMForSingleDigit.componentCovarianceMatrices[component]) + 0.0000001
                    covMatrixInv = np.linalg.pinv(GMMForSingleDigit.componentCovarianceMatrices[component])
                    xMinusCenter = np.subtract(x,GMMForSingleDigit.componentCenters[component])
                    xMinusCenter = xMinusCenter.reshape(1,64)
                    te = np.dot(xMinusCenter, np.dot(covMatrixInv, np.transpose(xMinusCenter)))
                    #Note we are not calculating  N(mean, variance) completely, we are interested only in part which varies between digits
                    probabilityMeasure += (1.0/math.sqrt(det)) * math.exp(-0.5 * te[0][0])
                if probabilityMeasure > maxProbabilityMeasure:
                    maxProbabilityMeasure = probabilityMeasure
                    predictedDigit = digit

            if predictedDigit == actualDigit:
                correctlyPredicted += 1
                correctlyPredictedPerDigit[predictedDigit] +=1

            countPerDigit[actualDigit] += 1

        accuracy = float(correctlyPredicted)/numObservations
        print("Accuracy of the GMM is ", accuracy * 100, " precentage")

        accuracyPerDigit = []
        for digit in range(10):
            accuracy = (float(correctlyPredictedPerDigit[digit])/countPerDigit[digit]) * 100
            print("Accuracy of the GMM for digit ", digit, " is ", accuracy," precentage")

def main():
    # Need to parse the command to obtain the respective arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--components', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)

    arguments        = vars(parser.parse_args())
    componentsInGMM  = arguments['components']
    trainingDataPath = arguments['train']
    testDataPath     = arguments['test']

    file = open(trainingDataPath, 'r')
    inpData = list(csv.reader(file, delimiter=','))
    file.close()
    trainingData = np.array(inpData, dtype=float)
    numObservations = trainingData.shape[0]
    numFeatures = trainingData.shape[1] - 1

    # We will split the data as feature vectors X and labels Y
    X = trainingData[:,:-1]
    Y = trainingData[:,-1]

    gaussianMixtureModelForAllDigits = GMMForAllDigits(componentsInGMM)
    gaussianMixtureModelForAllDigits.fit(X,Y)


    file = open(testDataPath, 'r')
    inpTestData = list(csv.reader(file, delimiter=','))
    file.close()
    testData = np.array(inpTestData, dtype=float)
    numObservations = testData.shape[0]
    numFeatures = testData.shape[1] - 1

    # We will split the data as feature vectors X and labels Y
    X = testData[:,:-1]
    Y = testData[:,-1]

    gaussianMixtureModelForAllDigits.predict(X,Y)


if __name__ == "__main__":
    main()