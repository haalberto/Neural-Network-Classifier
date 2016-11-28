# -*- coding: utf-8 -*-
"""
This script is an implementation of a classifier neural network with one hidden
layer.
It attemps to be as general as possible and easy to extend in functionality. It
includes a set of functions for making the network work.


Created on Sat Jul 23 16:54:31 2016

@author: Alberto Hinojosa
"""

import numpy as np
from scipy.optimize import minimize
#import pandas as pd

class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, num_labels, lamb):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_labels=num_labels
        self.lamb=lamb
        if self.num_labels==2:
            self.num_outputs=1
        else:
            self.num_outputs=self.num_labels
        self.num_params=self.hidden_size*(self.input_size+1)+self.num_outputs*(self.hidden_size+1)
        self.nn_params=np.zeros(self.num_params)
        
    def sigmoid(self, x):
        """Compute sigmoid function."""
        return 1./(1.+np.exp(-x))
    
    def sigmoidGradient(self, z):
        """Compute derivative of sigmoid function"""
        sig=self.sigmoid(z)
        return sig*(np.ones(z.shape)-sig)
    
    def multiclass_label_matrix(self, y):
        """
        Create a matrix of multiclass (vector) labels.
        
        Parameters
        num_labels: The number of different labels (>2)
        y: A 1d array of integers representing class labels. Each integer should
            be in the range [1,num_labels].
            
        Each value in y is mapped to row
            
        """
        if self.num_outputs>2:
            Y=np.zeros((y.size,self.num_outputs))
            for x in range(0,y.size):
                Y[x,y[x]-1]=1
        else:
            Y=y.copy()
            Y=Y.reshape(y.size,1)
        return Y
        
    def initialize_params (self):
        """ Pick random initial values for neural network parameters."""
        # Theta1 is hidden_size by (input_size+1)
        # Theta2 is num_outputs by (hidden_size+1)
                
        # Seed (uncomment for debugging)
        #np.random.seed(42)
        # Set interval length e. The interval for the numbers generated will be
        # [-e, e)
        e=1./np.sqrt(self.input_size+1)
        # Generate parameters
        num=self.hidden_size*(self.input_size+1)+self.num_outputs*(self.hidden_size+1)
        rands = (np.random.rand(num) -.5)*2*e
        return rands

    def unroll_parameters(self):
        """Reshape parameters as matrices."""
        Theta1=self.nn_params[0:self.hidden_size * (self.input_size + 1)].copy()
        Theta1=Theta1.reshape((self.hidden_size, self.input_size+1))
        Theta2=self.nn_params[self.hidden_size * (self.input_size + 1):].copy()
        Theta2=Theta2.reshape((self.num_outputs, self.hidden_size+1))
        return Theta1, Theta2
        # The copy is necessary because Theta1 and Theta2 might change later.

    def cost(self,nn_params, X, Y, lamb):
        """ Calculate the regularized cost function.
        
        Parameters
        nn_params: A 1d array containing the theta parameters. It needs to be
        unrolled as a matrix before being used.
        input_size: Number of inputs (without counting the bias)
        hidden_size: Number of elements in the hidden layer (without bias)
        num_labels: Number of labels or classes
        X: Matrix containing the training set features. Each example is a row.
        Y: Matrix containing the training set labels. Each examples is a row
            vector. Use the function 'multiclass_label_matrix' to generate it.
        lamb: Regularization factor
        
        """
        ### Step 1: Evaluate hypothesis function h for all examples
        ## Reshape parameters into matrix form
        #Theta1, Theta2 = self.unroll_parameters()
        Theta1=nn_params[0:self.hidden_size * (self.input_size + 1)].copy()
        Theta1=Theta1.reshape((self.hidden_size, self.input_size+1))
        Theta2=nn_params[self.hidden_size * (self.input_size + 1):].copy()
        Theta2=Theta2.reshape((self.num_outputs, self.hidden_size+1))
        # The copy is necessary because Theta1 and Theta2 change values later.
        
        ## Add column of ones to X.
        m = X.shape[0]; # Number of training examples.
        X = np.hstack((np.ones((m,1)), X))
        
        ## Evaluation of h
        a2=np.ones((self.hidden_size+1,m)) #Initialize activation functions
        z2=np.dot(Theta1,X.transpose()) #Because X' is like x with examples as columns.
        # Note: Each column of z2 corresponds to a training example.
        a2[1:,:]=self.sigmoid(z2) #Evaluate activation functions
        # Note: Each column of a2 is a vector with the values of the activation
        # functions for a given example.
        z3=np.dot(Theta2,a2)
        h=self.sigmoid(z3) #Hypothesis function for all examples (as columns)
        
        ### Step 2: Evaluation of (unregularized) cost
        # If there are more than two labels/classes, y needs to be
        # mapped from a scalar to a vector.
        
        # Cost
        J=-np.sum( Y.transpose()*np.log(h) )
        J=J-np.sum( (np.ones((self.num_outputs,m))-Y.transpose())*np.log(np.ones((self.num_outputs,m))-h) )
        
        ### Step 3: Regularization
        # Remove first column from Theta1 and Theta2, then square
        Theta1s=Theta1[:,1:]**2
        Theta2s=Theta2[:,1:]**2    
        # Regularized cost
        J=J+lamb/2*(np.sum(Theta1s)+np.sum(Theta2s))
        J=J/m
        return J

    def cost_grad(self,nn_params, X, Y, lamb):
        """ Calculate the gradient of the regularized cost function.
        
        Return cost and gradient as a tuple.
        
        Parameters
        nn_params: A 1d array containing the theta parameters. It needs to be
        unrolled as a matrix before being used.
        input_size: Number of inputs (without counting the bias)
        hidden_size: Number of elements in the hidden layer (without bias)
        num_labels: Number of labels or classes
        X: Matrix containing the training set features. Each example is a row.
        Y: Matrix containing the training set labels. Each examples is a row
            vector. Use the function 'multiclass_label_matrix' to generate it.
        lamb: Regularization factor
        
        """
        ### Step 1: Evaluate hypothesis function h for all examples
        ## Reshape parameters into matrix form
        #Theta1, Theta2 = self.unroll_parameters()
        Theta1=nn_params[0:self.hidden_size * (self.input_size + 1)].copy()
        Theta1=Theta1.reshape((self.hidden_size, self.input_size+1))
        Theta2=nn_params[self.hidden_size * (self.input_size + 1):].copy()
        Theta2=Theta2.reshape((self.num_outputs, self.hidden_size+1))
        # The copy is necessary because Theta1 and Theta2 change values later.
        
        ## Add column of ones to X. 
        m = X.shape[0]; # Number of training examples.
        X = np.hstack((np.ones((m,1)), X))
        
        ## Evaluation of h
        a2=np.ones((self.hidden_size+1,m)) #Initialize activation functions
        z2=np.dot(Theta1,X.transpose()) #Because X' is like x with examples as columns.
        # Note: Each column of z2 corresponds to a training example.
        a2[1:,:]=self.sigmoid(z2) #Evaluate activation functions
        # Note: Each column of a2 is a vector with the values of the activation
        # functions for a given example.
        z3=np.dot(Theta2,a2)
        h=self.sigmoid(z3) #Hypothesis function for all examples (as columns)
        
        ### Step 2: Compute gradients
        delta3= h-Y.transpose()
        # Remove first column of Theta2
        Theta2_short = Theta2[:,1:]
        delta2=np.dot(Theta2_short.transpose(), delta3) * self.sigmoidGradient(z2)
        # Calculate gradients for Theta2 and Theta1
        D2=np.dot(delta3,a2.transpose())
        D1=np.dot(delta2,X)
        
        ### Step 3: Regularization        
        # Set first column to zero for regularization purposes
        Theta1[:,0]=0
        Theta2[:,0]=0    
        # Regularized gradients
        D1=(D1+lamb*Theta1)/m
        D2=(D2+lamb*Theta2)/m
        
        # Unroll gradients
        gradJ=np.concatenate( (D1.reshape(D1.size),D2.reshape(D2.size)) )
        return gradJ
    
    def train(self, lamb, X, y, tol):
        """ Train network. """
        # Remap y
        Y = self.multiclass_label_matrix(y)
        # Initialize parameters
        self.nn_params=self.initialize_params()
        # Run optimizer
        result = minimize(self.cost, self.nn_params, args=(X,Y,lamb),
                                  method='BFGS', jac=self.cost_grad,
                                  options={'disp': True}, tol=tol)
        self.nn_params=result.x                          
        self.lamb=lamb
        
    def evaluate(self, X):
        """Evaluate model on data."""
        ## Reshape parameters into matrix form
        Theta1=self.nn_params[0:self.hidden_size * (self.input_size + 1)].copy()
        Theta1=Theta1.reshape((self.hidden_size, self.input_size+1))
        Theta2=self.nn_params[self.hidden_size * (self.input_size + 1):].copy()
        Theta2=Theta2.reshape((self.num_outputs, self.hidden_size+1))
        # The copy is necessary because Theta1 and Theta2 change values later.
        
        ## Add column of ones to X. 
        m = X.shape[0]; # Number of training examples.
        X = np.hstack((np.ones((m,1)), X))
        
        ## Evaluation of hypothesis h
        a2=np.ones((self.hidden_size+1,m)) #Initialize activation functions
        z2=np.dot(Theta1,X.transpose()) #Because X' is like x with examples as columns.
        # Note: Each column of z2 corresponds to a training example.
        a2[1:,:]=self.sigmoid(z2) #Evaluate activation functions
        # Note: Each column of a2 is a vector with the values of the activation
        # functions for a given example.
        z3=np.dot(Theta2,a2)
        return self.sigmoid(z3) #Hypothesis function for all examples (as columns)

    def predict(self,X):
        """Predict the class of data."""
        values=self.evaluate(X)
        if self.num_outputs ==1:
            values=values.reshape(values.size)
            pred=values>0.5
            pred=pred.astype(int)
        else:
            pred=np.argmax(values,0)+1
        return pred
    
    def accuracy(self,X,y):
        """Calculate accuracy of model on data."""
        predictions=self.predict(X)
        return np.mean((y==predictions).astype(int))
    
    def reg_exploration (self, lambdas, X, y, tol, X_val, y_val):
        """Calculate accuracy for different regularization constants."""
        # Create a temporary model to keep the present one unchanged.
        tempmodel = NeuralNetwork(self.input_size, self.hidden_size, self.num_labels, 0)
        #tempmodel=self.copy()
        accuracy=np.zeros(lambdas.size)
        for x in np.arange(lambdas.size):
            tempmodel.train(lambdas[x], X, y, tol)
            accuracy[x]=tempmodel.accuracy(X_val,y_val)
        return accuracy
            