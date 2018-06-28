
import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='cce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        activations: list of activation values for each layer
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        self.activations = []

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'cce':
            self.loss = CrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        hiddenActivation = "sigmoid"
        # Hidden layers
        for i in range(3):
            self.layers.append(LogisticLayer(128, 128, 
                               None, hiddenActivation, False))


        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10, 
                           None, outputActivation, True))

        
        self.inputWeights = inputWeights
        

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        self.activations = []
        for layer in self.layers[1:]:
            inp = layer.forward(inp)
            #add the 1 to output
            inp = np.insert(self._get_input_layer().outp,0,1,axis=0)
            self.activations.append(inp)
            
        
    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self._get_layer(target).activationDerivative(self.activations[target])
    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """

        #output_error = self._compute_error(-1)
        self._get_layer(-1).updateWeights(learningRate)
        for i in reversed(range(len(self.layers)-2)):
            self._get_layer(i).computeDerivative(self._compute_error(i+1), self._get_layer(i+1).weights)
            self._get_layer(i).updateWeights(learningRate)
            
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
            We can use the activations list to log the accuracy.
        """
        
        for i in range(self.epochs):
            #training sets are shuffled in mnist_seven.py, take random batch to train
            self._get_input_layer().forward(self.trainingSet.input[np.random.randint(0,len(self.trainingSet.input))])
            #insert 1 at start
            inp = np.insert(self._get_input_layer().outp,0,1,axis=0)
            self._feed_forward(inp)
            self._update_weights(self.learningRate)



    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        #self._get_input_layer().inp = test_instance
        outp = self._get_input_layer().forward(test_instance[0])
        outp = np.insert(outp,0,1,axis=0)
        self._feed_forward(outp)
        np.delete(self._get_output_layer().outp,0,axis=0)
        print ('Solution: ',self._get_output_layer().outp.argmax(axis=0),', true label: ', test_instance[1])
        #print (len(self._get_output_layer().outp))
        return self._get_output_layer().outp.argmax(axis=0)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = zip(self.testSet.input,self.testSet.label)
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
