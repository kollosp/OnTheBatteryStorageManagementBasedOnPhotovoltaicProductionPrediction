if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
from math import ceil
from datasets import utils
from sktimeSEAPF.Model import Model
from matplotlib import pyplot as plt
from utils.Plotter import Plotter
import seaborn as sns
from scipy import signal
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from scipy import signal
from scipy.special import expit
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MaxAbsScaler, PolynomialFeatures
from functools import reduce
from scipy.ndimage import gaussian_filter,grey_dilation,grey_erosion,morphological_gradient
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import sys
np.set_printoptions(threshold=sys.maxsize)



def onceKernel(shape, value = 1):
    return np.array([[value] * shape[1]] * shape[0])


class Blur(BaseEstimator,TransformerMixin):
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


class MyPipline(BaseEstimator,TransformerMixin):
    def __init__(self, methods):
        self.methods = methods

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.shape_ = X.shape
        self.steps_ = [("input", X.copy())]

        for m in self.methods:
            # make provided numbers of steps

            iterations = 1
            if len(m) > 2 and m[2] and m[2] > 1:
                iterations = m[2]

            for r in range(0, iterations):
                m[1].fit(X)
                X = m[1].transform(X)
                steps = None
                if type(X) is tuple:
                    steps = X[1]
                    X = X[0]

                if steps is None:
                    self.steps_.append((m[0] + (f"({r})" if iterations > 1 else ""), X.copy()))
                else:
                    for i in range(1, len(steps)): # first step is input (copy of provided argument). avoid adding repeating heatmaps to steps
                        self.steps_.append((m[0] + (f"({r})" if iterations > 1 else "") +"."+ steps[i][0], steps[i][1].copy()))

        return X, self.steps_

    def statistics(self):
        return "Layers:" + "\n".join([step[0] for step in self.steps_])


    def plot(self, rows = 3):
        l = len(self.steps_)
        count = sum(name[0] != "_" for name, _ in self.steps_)

        axis = ceil(count / rows ), rows
        fig, ax = plt.subplots(axis[1], axis[0])

        _ax = [ax[i, j] for j in range(axis[0]) for i in range(axis[1])]

        i = 0
        for (name, result) in self.steps_:
            if name[0]!="_":
                _ax[i].set(title=f"{i+1}: {name}")
                sns.heatmap(result.reshape(self.shape_), cmap='viridis', ax=_ax[i])
                _ax[i].invert_yaxis()
                i += 1
        return fig, ax

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

        # c = np.concatenate((X,y,w), axis=1)
        # print({"X":X, "y":c[:,1], "sample_weight":w})

        # add more observations to replace sample_weight
        # f = X.flatten()
        # print(np.sum(f > 0))

        # print_arr = np.concatenate([
        #    np.array([c[0, i], c[1, i]]).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
        # ])
        # print(print_arr)

        X = np.concatenate([
           # np.array([c[0, i], c[1, i]] * (c[2, i] * 10).astype(int)).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
           np.array([c[0, i], c[1, i]] ).reshape(-1,2) for i in range(c.shape[1]) if c[2, i] > 0
        ])
        # print(X.shape)


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

def f(heatmap):



    shape = heatmap.shape

    inner_pipe = MyPipline([
        ('_scaler', StandardScaler()),
        ('scaler', MaxAbsScaler()),

        ('BinaryMasked', FunctionTransformer(lambda x: np.where(x > x.mean(), 1, 0) * x)),

        ('conv2D', Blur(shape=shape, kernel=np.ones((3, 3)))),
        ('Magnitude0', Magnitude(gamma=2)),
    ])

    pipe = MyPipline([
        #('conv2D', Blur(shape=shape, kernel=onceKernel(shape=[3, 3], value=3))),
        # initial processing block
        ('conv2D', Blur(shape=shape, kernel=np.ones((17,1)))),
        ('conv2D', Blur(shape=shape, kernel=np.ones((1,5)))),
        ('_flattened0', FunctionTransformer(lambda x: x.reshape(-1, 1))),
        ('_scaler', StandardScaler()),
        ('scaler', MaxAbsScaler()),
        ('Magnitude0', Magnitude(gamma=2)),
        ('Sharpen', Blur(shape=shape, kernel=np.array([
            [0, -1, 0],
            [-1, 1, -1],
            [0, -1, 0],
        ]))),
        ('abs', FunctionTransformer(lambda x: x ** 2)),
        ##############################################################

        # repeated block
        # repeat
        ('inner_pipe', inner_pipe, 3),

        # conclusion block
        ('_scaler', StandardScaler()),
        ('scaler', MaxAbsScaler()),

        ('threshold', FunctionTransformer(lambda x: np.where(x > x.mean(), 1, 0))),

        ('_rebuild', FunctionTransformer(lambda x: x.reshape(-1, shape[-1]))),
        ('Regressor', RegressorTransformer(regressor=make_pipeline(
            PolynomialFeatures(11),
            LinearRegression()
        )))

        # ('Regressor', RegressorTransformer(regressor=SVR(C=10,epsilon=0.2, kernel='poly', degree=11)))
    ])

    heatmap, _ = pipe.fit_transform(heatmap)
    pipe.plot()
    print(pipe.statistics())

    print("mean, std: ", heatmap.mean(), heatmap.std(), reduce(lambda x,y:x*y, shape, 1))

    # sns.heatmap(heatmap, cmap='Reds', ax=ax[1])




"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    data, ts = utils.load_pv    (convert_index_to_time=True)
    print("Data len:", len(data) / 288)
    iterations = 3
    train_test_split = 288*160
    for i in range(iterations):
        y_train = data["Production"][i * train_test_split:(i + 1) * train_test_split]
        model = Model(latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=90,
                     y_bins=90, bandwidth=0.4, zeros_filter_modifier = 0, density_filter_modifier = 0)
        model.fit(y=y_train)
        #heatmap processing
        heatmap = model.overlay.heatmap

        f(heatmap)

    plt.show()


pipe1 = MyPipline([
    #('conv2D', Blur(shape=shape, kernel=onceKernel(shape=[3, 3], value=3))),
    ('conv2D', Blur(shape=shape, kernel=np.array([
        [0,1,0],
        [1] * 3,
        [0, 1, 0]
    ]))),
    ('_flattened0', FunctionTransformer(lambda x: x.reshape(-1, 1))),
    ('_scaler', StandardScaler()),
    ('scaler', MaxAbsScaler()),

    ('Magnitude0', Magnitude(gamma=2)),

    ('_rebuild1', FunctionTransformer(lambda x: x.reshape(-1, shape[-1]))),

    ('Opening', FunctionTransformer(lambda x: opening(x, size=(3, 3)))),
    ('Opening', FunctionTransformer(lambda x: opening(x, size=(3, 3)))),

    # ('Grad', FunctionTransformer(lambda x: morphological_gradient(x, size=(7, 7)))),
    ('Grad', FunctionTransformer(np_gradient)),

    ('threshold', FunctionTransformer(lambda x: np.where(x < x.mean(), 0, x))),
    ('_flattened', FunctionTransformer(lambda x: x.reshape(-1, 1))),
    # ('Magnitude', Magnitude(gamma=0.5)),
    ('_rebuild', FunctionTransformer(lambda x: x.reshape(-1, shape[-1]))),
    ('GaussianBlur', FunctionTransformer(lambda x: gaussian_filter(x, sigma=1, radius=7))),
    ('Erosion1', FunctionTransformer(lambda x: grey_erosion(x, size=(3, 3)))),

    ('_flattened2', FunctionTransformer(lambda x: x.reshape(-1, 1))),
    ('Magnitude2', Magnitude(gamma=4)),

    ('_rebuild', FunctionTransformer(lambda x: x.reshape(-1, shape[-1]))),
    ('Erosion1', FunctionTransformer(lambda x: grey_erosion(x, size=(3, 3)))),

    ('threshold', FunctionTransformer(lambda x: np.where(x > x.mean() * 1.2, 1, 0))),

    # ('threshold', FunctionTransformer(lambda x: np.concatenate((np.zeros((13, x.shape[1])), x[13:]) , axis=0))),

    # quite informative model.
    ('Regressor', RegressorTransformer(regressor=make_pipeline(
        PolynomialFeatures(11),
        LinearRegression()
    )))

    # ('Regressor', RegressorTransformer(regressor=SVR(C=10,epsilon=0.2, kernel='poly', degree=11)))
])