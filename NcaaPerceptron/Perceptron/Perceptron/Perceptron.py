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
        None

    # trains the model based on the inputs and targets
    #
    # inputs        [in] - an array (matrix) of all the input data; each row is an instance of the features
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

        print cm
        print trace(cm) / sum(cm)
        

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


# main() code goes here
input_file = csv.DictReader(open("NCAAdata.csv"))
NCAAdata   = None
testData   = None

# create an array (matrix) from the data in input_file
for row in input_file:
    swapcolumns = randint(0, 1)      # 0 means leave the columns in order; 1 means swap first two columns with next two for randomization

    if not swapcolumns:
        newRow = array([[float(row["Winrate"]), int(row["Seed"]), float(row["Winrate2"]), int(row["Seed2"]), 1]])
    else:
        newRow = array([[float(row["Winrate2"]), int(row["Seed2"]), float(row["Winrate"]), int(row["Seed"]), 0]])

    if row["season"] != "R":
        if NCAAdata is None:
            NCAAdata = newRow
        else:
            NCAAdata = concatenate((NCAAdata, newRow), axis=0)
    else:
        if testData is None:
            testData = newRow
        else:
            testData = concatenate((testData, newRow), axis=0)

pcn = Perceptron(NCAAdata[:, 0:5], NCAAdata[:, 4:])
pcn.Train(NCAAdata[:, 0:5], NCAAdata[:, 4:], 0.25, 10)
pcn.ConfusionMatrix(testData[:, 0:5], testData[:, 4:])
