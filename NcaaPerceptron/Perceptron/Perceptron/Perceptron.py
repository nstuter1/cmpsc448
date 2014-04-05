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
from random import shuffle
from numpy import *
import csv

# Perceptron ------------------------------------------------------------------------------------------
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
    # inputs         [in] - array (matrix) of all the input data; each row is an instance of features
    # targets        [in] - the results that the perceptron should get
    # initLearnRate  [in] - initial learning rate of the perceptron
    # tuningConstant [in] - constant for deciding learning rate over time
    # numIterations  [in] - number of times to train the model on inputs
    def Train(self, inputs, targets, initLearnRate, tuningConstant, numIterations):
        # Add the inputs that match the bias node
        inputs = concatenate(( inputs, -ones((self.nData, 1)) ), axis=1)

        # Training
        change = range(self.nData)

        # initialize the learning rate
        learnRate = initLearnRate

        for n in range(numIterations):
            # decrease learning rate over time after the 1000th iteration
            if n > 1000:
                learnRate = tuningConstant / float(n)

            self.outputs  = self.Forward(inputs);
            self.weights += learnRate * dot(transpose(inputs), targets-self.outputs)
        
            # Randomise order of inputs
            random.shuffle(change)
            inputs  = inputs[change, :]
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
    # verbose [in] - boolean; true when output on the confusion matrix and accuracy should be printed
    #
    # return - the accuracy over the testing data
    def ConfusionMatrix(self, inputs, targets, verbose):
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

        if verbose:
            print "Confusion Matrix:"
            print cm
            print "Accuracy:"
            print "{0}".format(trace(cm) / sum(cm))

        return trace(cm) / sum(cm)
        
# RegressionModel -------------------------------------------------------------------------------------
class RegressionModel:
    # constructor to initialize the weights and get the number of records in the data
    #
    # inputs [in] - the input training data
    def __init__(self, inputs):
        self.numRecords = shape(inputs)[0]
        self.weights    = random.rand( shape(inputs)[1], 1 ) * 0.1 - 0.05

    # calculate the gradient over a batch of the inputs
    #
    # batch   [in] - portion of records to calculate the gradient over
    # targets [in] - the target of each record
    #
    # return - the gradient over batch
    def MiniBatchGradient(self, batch, targets):
        temp  = targets - dot(batch, self.weights)  # calc of paranthesis part of least squares
        temp2 = dot(batch.T, temp)                  # equivalent to multiplying temp across the rows of
                                                    #   batch and summing over the records

        #  divide by the # of records and multiply by -2 to complete the gradient
        return -2 * temp2 / self.numRecords

    # perform weight updates using the gradient of portions of inputs, and repeat the process for
    #   numIterations
    #
    # inputs         [in] - array (matrix) of all the input data; each row is an instance of features
    # targets        [in] - the results that the model should get
    # initLearnRate  [in] - initial learning rate of the model
    # tuningConstant [in] - constant for deciding learning rate over time
    # numIterations  [in] - number of times to train the model on inputs
    def GradientDescent(self, inputs, targets, initLearnRate, tuningConstant, numIterations):
        batchSize = 10                       # size of batches to run the gradient on
        change    = range(self.numRecords)   # array of indexes to inputs and targets to be shuffled to
                                             #   rearrange the order of the records

        learnRate = initLearnRate            # initialize the learning rate

        # update the weights based on the learning rate and the calculation the gradient over 10
        #   records at a time; permute the records and repeat for numIterations
        for i in range(numIterations):
            # decrease learning rate over time after the 1000th iteration
            if i > 1000:
                learnRate = tuningConstant / float(n)
 
            # update the weights by the gradient of 10 records at a time
            for n in range(batchSize, self.numRecords, batchSize):
                if n + batchSize <= self.numRecords:
                    self.weights -= learnRate * self.MiniBatchGradient(inputs[n-batchSize:n, :],
                                                                      targets[n-batchSize:n, :])
                else:# combine last few records together if there won't be an even batchSize at the end
                    self.weights -= learnRate * self.MiniBatchGradient(
                                                                inputs[n-batchSize:self.numRecords, :],
                                                               targets[n-batchSize:self.numRecords, :])

            # Randomise order of the records in inputs
            random.shuffle(change)
            inputs  = inputs[change, :]
            targets = targets[change, :]

        #return self.weights

    # test the model on testing data and calculate the accuracy
    #
    # inputs  [in] - array (matrix) of all the input test data; each row is an instance of features
    # targets [in] - the results that the model should get
    # verbose [in] - boolean; true when output of accuracy should be printed
    #
    # return - the accuracy over the testing data
    def TestModel(self, inputs, targets, verbose):
        temp     = targets - dot(inputs, self.weights)   # calc of paranthesis part of least squares
        temp     = square(temp)                          # calculate the square of the paranthesis
        accuracy = 1 - (sum(temp, axis=0) / targets.shape[0])[0] # sum temp and divide by # of records;
                                                                 #   subtract from one for accuracy

        if verbose:
            print "Accuracy:"
            print "{0}".format(accuracy)

        return accuracy

# Global Functions-------------------------------------------------------------------------------------

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

# initialize, train, then test the data using a perceptron
#
# trainData      [in] - the data to train on
# testData       [in] - the data to test on
# initLearnRate  [in] - initial learning rate of the perceptron
# tuningConstant [in] - constant for deciding learning rate over time
# numTrains      [in] - the number of times to train the perceptron on trainData
# verbose        [in] - boolean; true when output of confusion matrix and accuracy should be printed
#
# return - the accuracy of the perceptron on testData
def RunPerceptron(trainData, testData, initLearnRate, tuningConstant, numTrains, verbose):
    targetCol = trainData.shape[1] - 1     # the column of the data that holds the target

    pcn = Perceptron(trainData[:, 0:targetCol], trainData[:, targetCol:])
    pcn.Train(trainData[:, 0:targetCol], trainData[:, targetCol:], initLearnRate, tuningConstant,
                                                                                        numTrains)
    return pcn.ConfusionMatrix(testData[:, 0:targetCol], testData[:, targetCol:], verbose)

# initialize, train, then test the data using a regression model
#
# trainData      [in] - the data to train on
# testData       [in] - the data to test on
# initLearnRate  [in] - initial learning rate of the perceptron
# tuningConstant [in] - constant for deciding learning rate over time
# numTrains      [in] - the number of times to train the perceptron on trainData
# verbose        [in] - boolean; true when output of accuracy should be printed
#
# return - the accuracy of the perceptron on testData
def RunRegression(trainData, testData, initLearnRate, tuningConstant, numTrains, verbose):
    targetCol = trainData.shape[1] - 1     # the column of the data that holds the target

    reg = RegressionModel(trainData[:, 0:targetCol])
    reg.GradientDescent(trainData[:, 0:targetCol], trainData[:, targetCol:], initLearnRate,
                                                                  tuningConstant, numTrains)
    return reg.TestModel(testData[:, 0:targetCol], testData[:, targetCol:], verbose)

# main() ----------------------------------------------------------------------------------------------

# set to true if you want to use the permutation test
usePermute = False

# set to true if you want to use regression with gradient descent
useRegression = True

seasonData = GetData("NCAAdata.csv")    # dictionary indexed by season letter; each season is a matrix
                                        #   of the data for that season
initLearnRate  = 0.1                    # initial learning rate of the perceptron
tuningConstant = 100                    # constant for deciding learning rate over time
numTrains      = 10000                  # number of times to run through the training data
numPermute     = 1000                   # number of times to run permutation test
totalAccuracy  = 0.0                    # total accuracy over all the runs of k-fold cross-validation
totalPValue    = 0.0                    # total p-value over every run of the permutation test
seasons = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R"]

if useRegression:
    print "Using a regression model with gradient descent:"
else:
    print "Using a perceptron model:"

# k-fold cross-validation, where every season is used as the testing data once
for i in seasons:
    print "\nRun with season {0} as the test season".format(i)

    trainData, testData = GetTrainAndTestData(seasonData, i)

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
