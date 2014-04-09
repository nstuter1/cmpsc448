# Created by: Alexander Anderson
# Class:      CMPSC 448 (Machine Learning)
# Date:       April 2014

from random import shuffle
from numpy import *

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
            print "Least-Square-Error Accuracy:"
            print "{0}".format(accuracy)

        return accuracy

    # creates the confusion matrix based on the testing data
    #
    # inputs  [in] - an array (matrix) of all the testing data; each row is an instance of the features
    # targets [in] - the results that the perceptron should get
    # verbose [in] - boolean; true when output on the confusion matrix and accuracy should be printed
    #
    # return - the accuracy over the testing data
    def ConfusionMatrix(self, inputs, targets, verbose):
        outputs = dot(inputs, self.weights)
    
        nClasses = shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = where(outputs > 0.5, 1, 0)
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
            print "Confusion Matrix Accuracy:"
            print "{0}".format(trace(cm) / sum(cm))

        return trace(cm) / sum(cm)

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
    reg.ConfusionMatrix(testData[:, 0:targetCol], testData[:, targetCol:], verbose)
    return reg.TestModel(testData[:, 0:targetCol], testData[:, targetCol:], verbose)