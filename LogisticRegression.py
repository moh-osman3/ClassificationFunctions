import numpy as np
import matplotlib.pyplot as plt

## Implementation of Sklearn LogisticRegression library 

class LogisticRegression:
    def __init__(self, eta, lambda_parameter, beta=None, probs = None, loss = None):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.probs = probs
        self.loss = loss

    # softmax     
    def __softmax(self, x):
        x -= np.max(x)
        sm = (np.exp(x).T / np.sum(np.exp(x),axis=1)).T
        return sm
    
    # one hot encoding
    def __onehot(self, y):
        n_values = np.max(y) + 1
        y_enc = np.eye(n_values)[y]
        return y_enc
     
    # returns back result of regression using current betas
    def __find_probs(self, X):
        X = self.__add_constant(X)

        # use optimal betas to get prediction
        y_pred = np.dot(X, self.beta)

        # turn prediction into "probability"
        real_probs = np.array(self.__softmax(y_pred))
        self.probs = real_probs
        return (real_probs)
    
    # add intercept column
    def __add_constant(self, X):
        return np.vstack((np.ones(len(X)), X.T)).T

    # negative log loss, use to optimize betas
    def __log_loss(self, X, y):
        y_pred = self.__find_probs(X)
        loss = 0
        length = len(y_pred)
        for i in range(length):
            # check that the highest probability is the true class
            if np.argmax(y_pred) == np.argmax(y):
                loss += np.log(np.max(y_pred))
        loss /= length
        
        return loss
        
    # fit function
    def fit(self, X, y): 
        y = self.__onehot(y)
        x_new = self.__add_constant(X)
        betas = np.array([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]])

        loss_list = [] 
        self.beta = betas
        # gradient descent
        for i in range(30000):
            preds = self.__find_probs(X)
            probs = list(map(np.max, preds))

            # compute gradient
            dif = preds - y
            gradient = np.dot(x_new.T, dif) / y.shape[0] + 2 * self.lambda_parameter * self.beta
            self.beta -= self.eta * gradient
            loss_list.append(self.__loss(y, preds))

        self.loss = loss_list
        
    # make prediction
    def predict(self, X_pred):
        # turn prediction into "probability"
        real_probs = self.__find_probs(X_pred)
        self.probs = real_probs

        real_pred = []
        for row in real_probs:
            real_pred.append(np.argmax(row))
         
        return np.array(real_pred)


    def __loss(self, y, y_pred):
        loss = []
        for i in range(y.shape[0]):
            loss.append(y[i] * np.log(y_pred[i]))
        non_reg = np.sum(loss)
        loss_reg = non_reg + self.lambda_parameter * np.sum(y_pred**2)
        loss_reg /= y.shape[0]

        return -loss_reg


    def visualize_loss(self, output_file, show_charts=False):
        fig, ax = plt.subplots(1, 1, figsize = (5, 5))
        ys = self.loss
        xs = range(1, len(self.loss) + 1)
        ax.grid(True, color='w', linestyle='-', linewidth=2)
        fig.gca().patch.set_facecolor('0.6')
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Loss")
        ax.set_title("3.2 : Loss versus Number of Iterations")
        if show_charts:
            ax.plot(xs, ys)
        
