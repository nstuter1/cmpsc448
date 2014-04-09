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
# Date:        April 2014

from random import shuffle
from numpy import *

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
