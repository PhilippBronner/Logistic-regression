import numpy as np 
import pandas as pd 

class LogisticRegression:
    
    def __init__(self, parameter):
        self.theta = np.ones((parameter)) #Fit parameters
        pass
        
    def fit(self, X, y, steps = 500):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
            steps (int): number of optimization steps that should be performed. Default =500.
        
        Returns:
            acc_obs (array<steps>): observation of the accurancy during the optimization.
        """
        c = 1 #optional parameter if a while loop with condition is used instead of for loop.
        m = len(X[:,0])
        n = len(X[0,:])+1
        Xnew = np.ones([m,n])
        Xnew[:,1:] = X
        X = Xnew
        acc_obs = np.zeros(steps) #dummy array for the accurancy observation during the steps.
        for ii in range(steps):
            h_theta = gradient(self.theta, X) #calculate the gradient
            diff = y - h_theta
            correct = np.sum(np.transpose(X)*diff, axis = 1)
            c = sum(abs(correct)) #optional prameter for training with a while loop,
                                  #instead of a fixt number in a foor loop.
            alpha = 0.005 #step size
            self.theta = self.theta + correct*alpha
            ##Accuracy observation
            y_pred = self.predict(X[:,1:])
            acc_obs[ii] = binary_accuracy(y_true=y, y_pred=y_pred)
        return acc_obs
    
    
    def predict(self, X):
        """
        Generates predictions. Clculates a values [0 1] for each data point,
        whether it is 0 or 1.
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        z = np.sum(X*self.theta[1:], axis = 1)+self.theta[0]
        y = sigmoid(z)
        return y
        

        
# --- Some utility functions 

def gradient(theta,x):
    """
    Calculates the gradient from a given theta and X data points.
    
    Args:
        theta (array<n>): features of the regression
        X (array<m,n>): a matrix of floats with 
            m rows (#samples) and n columns (#features)
    
    Returns:
        A length m array of floats in the range [0, 1]
        with probability-like predictions
    """
    z = np.sum(x*theta, axis = 1)
    h_theta = sigmoid(z)
    return h_theta

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

        