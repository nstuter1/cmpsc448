# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)
#
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.
#
# Stephen Marsland, 2008

# Modified by: Alexander Anderson
# Class:       CMPSC 448 (Machine Learning)
# Date:        February 2014

from random import randint
from numpy import *
import csv

class Perceptron:
    # the constructor to initialize the weights and set up the perceptron
    #
    # inputs  [in] - an array (matrix) of all the input data; each row is an instance of the features
    # targets [in] - the results that the perceptron should get
    def __init__(self, inputs, targets):
        # Set up network size
        if ndim(inputs) > 1:
            self.nIn = shape(inputs)[1]
        else: 
            self.nIn = 1
    
        if ndim(targets) > 1:
            self.nOut = shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = shape(inputs)[0]
    
        # Initialise network
        self.weights = random.rand(self.nIn+1, self.nOut) * 0.1 - 0.05

    # trains the model based on the inputs and targets
    #
    # inputs        [in] - an array (matrix) of all the input data; each row is an instance of features
    # targets       [in] - the results that the perceptron should get
    # learnRate     [in] - the degree which weights should change on training
    # numIterations [in] - number of times to train the model on inputs
    def Train(self, inputs, targets, learnRate, numIterations):
        # Add the inputs that match the bias node
        inputs = concatenate(( inputs, -ones((self.nData, 1)) ), axis=1)

        # Training
        change = range(self.nData)

        for n in range(numIterations):
            self.outputs = self.Forward(inputs);
            self.weights += learnRate * dot(transpose(inputs), targets-self.outputs)
        
            # Randomise order of inputs
            random.shuffle(change)
            inputs = inputs[change, :]
            targets = targets[change, :]
            
        #return self.weights

    # calculates if the neuron would fire based on the inputs and the previously calculated weights
    #
    # inputs  [in] - an array (matrix) of all the input data; each row is an instance of the features
    def Forward(self, inputs):
        outputs = dot(inputs, self.weights)

        # Threshold the outputs
        return where(outputs > 0, 1, 0)

    # creates the confusion matrix based on the testing data
    #
    # inputs  [in] - an array (matrix) of all the testing data; each row is an instance of the features
    # targets [in] - the results that the perceptron should get
    def ConfusionMatrix(self, inputs, targets):
        # Add the inputs that match the bias node
        inputs = concatenate(( inputs, -ones((shape(inputs)[0], 1)) ), axis=1)
        
        outputs = dot(inputs, self.weights)
    
        nClasses = shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = argmax(outputs, 1)
            targets = argmax(targets, 1)

        cm = zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = sum(where(outputs == i, 1, 0) * where(targets == j, 1, 0))

        print "Confusion Matrix:"
        print cm
        print "Error:"
        print "{0}\n".format(trace(cm) / sum(cm))

        return trace(cm) / sum(cm)
        

# read in the data from file and separate it into a dictionary of matrices of the data for each season
#
# file [in] - a string specifying the .csv file to read the data from
#
# return - a dictionary of matrices of the data for each season, where the season letter is the key
def GetData(file):
    input_file = csv.DictReader(open(file))    # pointer to the open file
    seasonData = dict()                        # the dictionary of season data that will be returned

    # add to seasonData from the data in input_file
    for row in input_file:
        swapColumns = randint(0, 1)   # 0 means leave the columns in order; 1 means swap first two
                                      #   columns with next two for randomization

        # swap columns based on swapColumns
        if not swapColumns:
            newRow = array([[float(row["Winrate"]), int(row["Seed"]), float(row["Winrate2"]),
                                                                           int(row["Seed2"]), 1]])
        else:
            newRow = array([[float(row["Winrate2"]), int(row["Seed2"]), float(row["Winrate"]),
                                                                             int(row["Seed"]), 0]])

        # add newRow to seasonData based on the season the row pertains to
        if row["season"] not in seasonData:
            seasonData[row["season"]] = newRow
        else:
            seasonData[row["season"]] = concatenate((seasonData[row["season"]], newRow), axis=0)

    return seasonData

# gets the testing and training data, where testSeason is the season to use as the testing data, and
#   the rest of the seasons are used for the training data
#
# data       [in] = dictionary of the data for each season, indexed by the season letter
# testSeason [in] = char of the season to use as the testing data; key into data
# 
# return - a tuple where the first element is a matrix of the training data, and the second element is
#            a matrix of the testing data
def GetTrainAndTestData(data, testSeason):
    trainData = None                # matrix of the seasons to train on
    testData  = data[testSeason]    # matrix of the seasons to test on

    # run through every key of data and concatenate the matrices of each season that isn't testSeason;
    #   together, they make up the training data
    for key in data:
        if key != testSeason:            # key is not the testSeason
            if trainData is None:        # trainData is empty
                trainData = data[key]
            else:                        # trainData already has some data so concatenate
                trainData = concatenate((trainData, data[key]), axis=0)

    return trainData, testData


# Example with AND and XOR logic functions
"""
a = array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
b = array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

p = self.Perceptron(a[:, 0:2], a[:, 2:])
p.Train(a[:, 0:2], a[:, 2:], 0.25, 10)
p.ConfusionMatrix(a[:, 0:2], a[:, 2:])

q = self.Perceptron(a[:, 0:2], b[:, 2:])
q.Train(a[:, 0:2], b[:, 2:], 0.25, 10)
q.ConfusionMatrix(a[:, 0:2], b[:, 2:])
"""


#######################
# main() code goes here
#######################

seasonData = GetData("NCAAdata.csv")    # dictionary indexed by season letter; each season is a matrix
                                        #   of the data for that season
learnRate  = 0.1                        # learning rate of the perceptron
numTrains  = 1000                       # number of times to run through the training data
totalError = 0                          # total error over all the runs of the k-fold cross-validation
numBetter  = 0                          # number of times permutation test did better than the original
numPermute = 10000                      # number of times to run permutation test
seasons    = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R"]

# k-fold cross-validation, where every season is used as the testing data once
for i in seasons:
    print "Run with season {0} as the test season".format(i)

    trainData, testData = GetTrainAndTestData(seasonData, i)

    # initialize, train, then test the data
    pcn = Perceptron(trainData[:, 0:4], trainData[:, 4:])
    pcn.Train(trainData[:, 0:4], trainData[:, 4:], learnRate, numTrains)
    totalError += pcn.ConfusionMatrix(testData[:, 0:4], testData[:, 4:])

    # run permutation test, where we permute target data and compare the error to the real perceptron
    for j in range(numPermute):
        break

print "Average error over the k-fold cross-validation:"
print totalError / float(len(seasons))
