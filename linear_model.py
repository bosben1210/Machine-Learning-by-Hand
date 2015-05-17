import numpy as np

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



