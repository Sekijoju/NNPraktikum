
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
                 loss='bce', learningRate=0.01, epochs=50):

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
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'crossentropy':
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

        # Hidden layers
        hiddenActivation = "sigmoid"
        self.layers.append(LogisticLayer(128, 64, 
                           None, hiddenActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(64, 10, 
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
        nextInput = inp
        for layer in self.layers:
            # The output of the current layer is the input of the next layer:
            nextInput = layer.forward(nextInput)
            # Add bias "1" at the beginning of the next input:
            nextInput = np.insert(nextInput, 0, 1, axis=0)
        
    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateError(target, self._get_output_layer().outp)
    
    def _update_weights(self, learningRate, label):
        """
        Update the weights of the layers by propagating back the error
        """
        # Backpropagation of error:
        # To calculate the derivatives, we iterate over the layers in reverse order:
        # In case of the output layer, next_weights is array of 1
        # and next_derivatives - the derivative of the error will be the errors
        target = np.zeros(self._get_output_layer().nOut)
        target[label] = 1.0
        next_derivative = self.loss.calculateDerivative(target, self._get_output_layer().outp)
        next_layer_weights = np.identity(self._get_output_layer().nOut)

        # Backpropagate:
        for layer in reversed(self.layers):
            # Compute the derivatives:
            next_derivative = layer.computeDerivative(next_derivative, next_layer_weights.T)
            # Remove bias from weights, so it matches the output size of the next layer:
            next_layer_weights = np.delete(layer.weights, 0, axis=0)

        # Then update the weights:
        for layer in self.layers:
            layer.updateWeights(learningRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))
                #print("Learning rate: {0:.4f}".format(self.learningRate))

            for input, label in zip(self.trainingSet.input, self.trainingSet.label):
                # Compute the network output via feed forward:
                self._feed_forward(input)
                # Backpropagate the error and update the weights
                self._update_weights(self.learningRate, label)

            # Determine accuracy by evaluating the validation set:
            accuracy = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet))
            # Record the performance of each epoch for later usages
            # e.g. plotting, reporting..
            self.performances.append(accuracy)

            if verbose:
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

            #self.learningRate = self.learningRate * 0.9


    def classify(self, test_instance):
        """Classify a single instance.

        Parameters
        ----------
        test_instance : (list of floats, label)

        Returns
        -------
        int :
            The recognized digit (0-9).
        """
        # Classify an instance given the model of the classifier
        # Compute the network output via feed forward:
        input = test_instance[0]
        #label = test_instance[1]
        self._feed_forward(input)
        output = self._get_output_layer().outp
        # Pick the digit with highest probability:
        result = np.argmax(output, axis=0)
        #print ('Solution: ', result,', true label: ', label)
        return result

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
            test = self.testSet
        # Zip with labels, for debugging purposes:
        test = zip(test.input, test.label)
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
