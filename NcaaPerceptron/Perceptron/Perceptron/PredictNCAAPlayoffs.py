# Created by: Alexander Anderson
# Class:      CMPSC 448 (Machine Learning)
# Date:       April 2014

from random import randint
from numpy import *
import csv

from Perceptron import *
from Regression import *

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
            newRow = array([[float(row["Winrate"]), int(row["Seed"]), float(row["Oppwinrate"]),
                             float(row["Winrate2"]), int(row["Seed2"]), float(row["Oppwinrate2"]), 1]])
        else:
            newRow = array([[float(row["Winrate2"]), int(row["Seed2"]), float(row["Oppwinrate2"]),
                             float(row["Winrate"]), int(row["Seed"]), float(row["Oppwinrate"]), 0]])

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



# main() ----------------------------------------------------------------------------------------------

# set to true if you want to use the permutation test
usePermute = False

# set to true if you want to use regression with gradient descent
useRegression = False

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
