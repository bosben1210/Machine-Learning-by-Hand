import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression(object):

	def __init__(self, fit_intercept=True, normalize=False):
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.params_ = None

	def fit(self, X, y):
		if self.fit_intercept:
			X = np.hstack((np.ones((len(X),1)), X))
		self.params_ = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

	def predict(self, X):
		if self.fit_intercept:
			X = np.hstack((np.ones((len(X),1)), X))
		return np.dot(X, self.params_)

	def score(self, X, y):
		y_predict = self.predict(X)
		y_mean = np.mean(y)
		TSS = np.sum((y_mean - y) ** 2)
		RSS = np.sum((y_predict - y) ** 2)
		return 1 - RSS / TSS

class Ridge_Regression(Linear_Regression):

	def __init__(self, fit_intercept=True, normalize=False, lamb=0):
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.params_ = None
		self.lamb = lamb

	def fit(self, X, y):
		if self.fit_intercept:
			X = np.hstack((np.ones((len(X),1)), X))
			eye = np.eye(X.shape[1]) 
			eye[0,:] = 0
		else:
			eye = np.eye(X.shape[1]) * self.lamb
		self.params_ = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + eye), X.T), y)

	def predict(self, X):
		if self.fit_intercept:
			X = np.hstack((np.ones((len(X),1)), X))
		return np.dot(X, self.params_)

	def score(self, X, y):
		y_predict = self.predict(X)
		y_mean = np.mean(y)
		TSS = np.sum((y_mean - y) ** 2)
		RSS = np.sum((y_predict - y) ** 2)
		return 1 - RSS / TSS


class Logistic_Regression(object):

    def __init__(self, alpha=0.01, num_iter=100000, optimizer=None):
        self.alpha = alpha
        self.num_iter = num_iter
        self.optimizer = optimizer
        
    
    def fit(self, X, y):
        '''
        Fit the model according to the given training data
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
            
        Returns
        -------
        self : object
            Returns self
            
        Notes :
        -------
        Regularization term and optimizer not included at this moment
        '''
    
        m = len(y)
        X_design = np.c_[np.ones(m), X] 
        theta = np.random.randn(X_design.shape[1], 1)
        cost_list = []
        
        for i in xrange(self.num_iter):
            error = self.__sigmoid(np.dot(X_design, theta)) - y
            theta -= np.dot(error.T, X_design).T / float(m) * self.alpha
            cost = self.__cost_func(theta, X_design, y)
            cost_list.append(cost)

        self.theta_opt = theta
        self.cost_list = cost_list
        return self
    
    def predict(self, X):
        '''
        Predict new output using the model given training data
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and n_features is the number of features.
            
        Returns
        -------
        z : array-like, shape (n_samples, 1)
            Returns predicted outputs
            
        Notes :
        -------
        Regularization term not included at this moment
        '''
        
        X_design = np.c_[np.ones(len(X)), X]
        z = np.round(self.__sigmoid(np.dot(X_design, self.theta_opt)))
        return z
    
    
    def score(self, X, y):
        '''
        Returns accuracy of prediction
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
            
        Returns
        -------
        score : float
            percentage of correct prediction
            
        Notes :
        -------
        
        '''
        return np.mean(self.predict(X) == y)
    
    def cost_plot(self):
        '''
        Plot the change of cost function over iterations
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Plot of value of cost function
        '''
        plt.plot(self.cost_list)
    
    def __sigmoid(self, z):
        '''
        apply sigmoid function to input

        Parameters :
        ------------
        z : array-like
            input of sigmoid function
        Returns : 
        ---------
        output value

        Notes :
        -------

        '''
        return 1.0/(1.0 + np.exp(- 1.0*z))
    
    
    def __cost_func(self, theta, X, y):
        '''
        Calculate the cost function of logistic regression

        Parameters :
        ------------
        theta : array-like, shape (n_parameters,)
                initial parameter of features, where n_features is the number of features.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector related to X. 

        Returns : 
        ---------
        J : float
            cost of logistic regression using the input

        Notes :
        -------
        Regularization term not included
        '''

        m = len(y)
        J = -1.0/m * sum(np.multiply(y, np.log(self.__sigmoid(np.dot(X, theta)))) + np.multiply(1-y, np.log(1-self.__sigmoid(np.dot(X, theta))))) 
        return J

if __name__ == '__main__':
    # a simple example
    
    X = np.random.rand(300,2)
    y = (np.atleast_2d(1.0/(1.0 + np.exp(- 1.0*((X[:,0] - .8) * 2 + (X[:,1]-.5))))).T >= 0.5).astype(int)
    
    clr = Logistic_Regression()
    clr.fit(X, y)
    clr.score(X, y)
    
    xx, yy = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
    Z = np.round(clr.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,8))
    plt.contourf(xx, yy, Z, alpha=.6)
    plt.scatter(X[:,0], X[:,1], c=y, s=50)
    plt.show()