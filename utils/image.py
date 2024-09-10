from sklearn.preprocessing import FunctionTransformer, StandardScaler, MaxAbsScaler, PolynomialFeatures
from functools import reduce
from scipy.ndimage import gaussian_filter,grey_dilation,grey_erosion,morphological_gradient
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

def onceKernel(shape, value = 1):
    return np.array([[value] * shape[1]] * shape[0])

class Convolution(BaseEstimator,TransformerMixin):
    def __init__(self, kernel, shape):
        self.kernel = kernel
        self.shape = shape

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        shape = X.shape
        if not shape == self.shape:
            X = X.reshape(self.shape)
        X = signal.convolve2d(X, self.kernel, boundary='symm', mode='same')
        return X.reshape(shape)

class Magnitude(BaseEstimator,TransformerMixin):
    def __init__(self, gamma=2):
        self.gamma = gamma

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return expit(4 * self.gamma * (X - X.mean()))



class RegressorTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        shape = X.shape
        c = np.concatenate([[[j], [i], [w]]
                            for i, x in enumerate(X)
                            for j, w in enumerate(x)], axis=1)

        X = np.concatenate([
           # np.array([c[0, i], c[1, i]] * (c[2, i] * 10).astype(int)).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
           np.array([c[0, i], c[1, i]] ).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
        ])

        # cls = SVR(C=self.C, epsilon=self.epsilon, degree=self.degree)
        self.regressor.fit(X=X[:, 0:1], y=X[:, 1])

        pred = self.regressor.predict(X=np.array(list(range(0, shape[1]))).reshape(-1, 1))
        pred_arr = np.zeros(shape)
        pred_i = pred.astype(int)
        pred_i[pred_i >= pred_arr.shape[1]] = pred_arr.shape[1] -1
        # print(pred)
        for i in range(0, pred_arr.shape[1]):
            # print(pred_arr.shape[1], i, pred_i[i])
            pred_arr[pred_i[i], i] = 1

        return pred_arr

class Analyse(BaseEstimator,TransformerMixin):
    def __init__(self, bins=100):
        self.bins = bins

    def get_threshold(self, percentage_left):
        """
        percentage_left: percentage values that should left in the histogram after threshold application
        """
        s = np.sum(self.value_histogram_)
        k = s

        for i in range(self.bins -1,0,-1):
            if s * percentage_left < k :
                k = k - self.value_histogram_[i] #remove last bin
            else:
                return self.value_histogram_bins_[i]

        return self.value_histogram_bins_[0]

    def fit(self, X, y=None):
        self.shape_ = X.shape
        self.X_ = X.copy()
        self.value_histogram_, self.value_histogram_bins_ = np.histogram(X.flatten(), bins=self.bins)
        self.along_x_ = np.sum(X, axis=1)
        self.along_y_ = np.sum(X, axis=0)
        return self

    def transform(self, X, y=None):
        return X

    def plot(self):
        fig, ax = plt.subplots(2,2)
        _ax = ax[0,0]

        sns.heatmap(self.X_ , cmap='viridis', ax=_ax)
        _ax.invert_yaxis()
        _ax = ax[1, 1]
        _ax.hist(self.value_histogram_bins_[:-1], self.value_histogram_bins_, weights=self.value_histogram_)
        _ax = ax[0, 1]
        s = list(range(self.X_.shape[0])) #+ [self.X_.shape[0]]
        _ax.barh(s, self.along_x_)

        _ax = ax[1, 0]
        s = list(range(self.X_.shape[1])) #+ [self.X_.shape[1]]
        _ax.bar(s, self.along_y_)
        return fig, ax


def opening(mat, size=(3,3)):
    mat = grey_erosion(mat, size=size)
    return grey_dilation(mat, size=size)

def closing(mat, size=(3,3)):
    mat =  grey_dilation(mat, size=size)
    return grey_erosion(mat, size=size)

def np_gradient(x):
    x = np.gradient(x)
    x = np.sqrt(x[0] ** 2 + x[1] ** 2)
    print(x)
    return x
