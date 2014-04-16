import csv
import numpy as np
from Perceptron import *
from Regression import *

data = np.loadtxt('clean_data.csv', delimiter = ',')    # getting the clean data from the file

print data

for x in data:                                          # data says how much it defaults
    if x[9] != 0:                                       # we just want whether it defaults
        x[9] = 1                                        # 0 if no default, 1 if default

k = np.array_split(data, 11)                            # this splits the data into 11 parts for k crossfold validation

# main() ----------------------------------------------------------------------------------------------

# set to true if you want to use the permutation test
usePermute = False

# set to true if you want to use regression with gradient descent
useRegression = False

initLearnRate  = 0.1                    # initial learning rate of the perceptron
tuningConstant = 100                    # constant for deciding learning rate over time
numTrains      = 10000                  # number of times to run through the training data
numPermute     = 1000                   # number of times to run permutation test
totalAccuracy  = 0.0                    # total accuracy over all the runs of k-fold cross-validation
totalPValue    = 0.0                    # total p-value over every run of the permutation test

if useRegression:
    print "Using a regression model with gradient descent:"
else:
    print "Using a perceptron model:"

# k-fold cross-validation, where every season is used as the testing data once
for i in xrange(0, len(k)):
    print "\nRun with k[{0}] as the test season".format(i)

    testData  = k[i]
    trainData = None

    for c in xrange(0, len(k)):
        if c != i:
            if trainData == None:
                trainData = k[c]
            else:
                trainData = np.concatenate((trainData, k[c]), axis=0)

    # initialize, train, then test the data
    if useRegression:
        origAccuracy = RunRegression(trainData, testData, initLearnRate, tuningConstant, numTrains,
                                                                                               True)
    else:
        origAccuracy = RunPerceptron(trainData, testData, initLearnRate, tuningConstant, numTrains,
                                                                                               True)

    totalAccuracy += origAccuracy

    # run permutation test, where we permute target data and compare accuracy to the real model
    if usePermute:
        numBetter = 0                   # number of times permutation test did better than the original

        print "Running permutation test and calculating p-value..."

        # permute the target column, run the model, and keep track of every time the permutation
        #   does better than the original data
        for j in range(numPermute):
            # permute the target column of the training and testing data
            shuffle(trainData[:, 4])
            shuffle(testData[:, 4])

            if useRegression:
                accuracy = RunRegression(trainData, testData, initLearnRate, tuningConstant, numTrains,
                                                                                                 False)
            else:
                accuracy = RunPerceptron(trainData, testData, initLearnRate, tuningConstant, numTrains,
                                                                                                 False)

            # track if permutation did better than the original data
            if accuracy >= origAccuracy:
                numBetter += 1

        # calculat the p-value
        pValue       = numBetter / float(numPermute)
        totalPValue += pValue

        print "p-value = {0}".format(pValue)

print "\nAverage accuracy over the k-fold cross-validation:"
print totalAccuracy / float(len(seasons))

if usePermute:
    print "Average p-value over the k-fold cross-validation:"
    print totalPValue / float(len(seasons))