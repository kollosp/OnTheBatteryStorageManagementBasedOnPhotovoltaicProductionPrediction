if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
from math import ceil
from datasets import utils
from sktimeSEAPF.Model import Model
from matplotlib import pyplot as plt
from utils.Plotter import Plotter
from utils.MyPipeline import MyPipline
from sklearn.pipeline import make_pipeline
import utils.image as image
from utils.RegressorTransformer import RegressorTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import seaborn as sns
from scipy import signal

from sklearn.tree import DecisionTreeRegressor
# from scipy import signal
# from scipy.special import expit
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MaxAbsScaler, PolynomialFeatures
from functools import reduce
# from scipy.ndimage import gaussian_filter,grey_dilation,grey_erosion,morphological_gradient
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import sys
# np.set_printoptions(threshold=sys.maxsize)

def make_h_line(shape):
    z = np.zeros(shape)
    z[shape[0]//2,:] = 1
    return z
def make_v_line(shape):
    z = np.zeros(shape)
    z[:, shape[1]//2] = 1
    return z

class HitPoints(BaseEstimator,TransformerMixin):
    def __init__(self, max_iter=3, neighbourhood=5):
        self.max_iter = max_iter
        self.neighbourhood = neighbourhood

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        if len(X.shape) < 2:
            raise ValueError("HitPoints.transform: X should have at least 2 dimensions ")

        Z = np.zeros(X.shape)

        for _ in range(self.max_iter):
            imx = np.argmax(X, axis=0)
            # print(list(imx))
            for j,index in enumerate(imx):
                Z[index, j] = 1
                X[index, j-self.neighbourhood:j+self.neighbourhood] = 0 # clear max


        return Z


class AdaptiveThreshold(BaseEstimator,TransformerMixin):
    def __init__(self, threshold=.8):
        self.threshold = threshold
    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        analyse = image.Analyse()
        analyse.fit(X)
        threshold = analyse.get_threshold(self.threshold)
        X[X < threshold] = 0
        X[X >= threshold] = 1

        return X



def f(heatmap):
    shape = heatmap.shape

    inner_pipe = MyPipline([
        ('_scaler', StandardScaler()),
        ('scaler', MaxAbsScaler()),

        ('BinaryMasked', FunctionTransformer(lambda x: np.where(x > x.mean(), 1, 0) * x)),

        ('conv2D', image.Convolution(shape=shape, kernel=np.ones((3, 3)))),
        ('Magnitude0', image.Magnitude(gamma=2)),
    ])

    pipe = MyPipline([
        #('conv2D', Blur(shape=shape, kernel=onceKernel(shape=[3, 3], value=3))),
        # initial processing block
        ('conv2D', image.Convolution(shape=shape, kernel=np.ones((17,1)))),
        ('conv2D', image.Convolution(shape=shape, kernel=np.ones((1,5)))),
        ('_flattened0', FunctionTransformer(lambda x: x.reshape(-1, 1))),
        ('_scaler', StandardScaler()),
        ('scaler', MaxAbsScaler()),
        ('Magnitude0',image.Magnitude(gamma=2)),
        ('Sharpen', image.Convolution(shape=shape, kernel=np.array([
            [0, -1, 0],
            [-1, 1, -1],
            [0, -1, 0],
        ]))),
        ('abs', FunctionTransformer(lambda x: x ** 2)),
        ('_rebuild1', FunctionTransformer(lambda x: x.reshape(-1, shape[-1]))),
        ('scaler', MaxAbsScaler()),

        ('adaptive', AdaptiveThreshold(0.8)),
        ('mul', FunctionTransformer(lambda x: pipe.get_step(title="input") * x) ),
        ('scaler', MaxAbsScaler()),
        # ('_rebuild1', FunctionTransformer(lambda x: make_v_line(x.shape))),
        ('Magnitude0', image.Magnitude(gamma=2)),
        # ('scaler', MaxAbsScaler()),
        ('conv2D', image.Convolution(shape=shape, kernel=np.ones((5, 5)))),
    ])

    pipe = MyPipline([
        # ('conv2D', image.Convolution(shape=shape, kernel=np.ones((1, 5)))),
        # ('_flattened0', FunctionTransformer(lambda x: x.reshape(-1, 1))),
        # ('_scaler', StandardScaler()),
        # ('scaler', MaxAbsScaler()),
        # ('Magnitude0', image.Magnitude(gamma=2)),
        # ('_rebuild1', FunctionTransformer(lambda x: x.reshape(-1, shape[-1]))),
        ('HitPoints', HitPoints(max_iter=3, neighbourhood=int(0.1 * shape[0])+1)),
        ('conv2D', image.Convolution(shape=shape, kernel=np.ones((1, int(0.01 * shape[0])+1)))),
        ('conv2D', image.Convolution(shape=shape, kernel=np.ones((shape[0] // 9 + 1, 1)))),
        #
        # ('HitPoints', HitPoints(max_iter=1, neighbourhood=int(0.1 * shape[0]) + 1)),
        # ('conv2D', image.Convolution(shape=shape, kernel=np.ones((1, int(0.05 * shape[0]) + 1)))),
        ('Regressor', RegressorTransformer(regressor=make_pipeline(
            PolynomialFeatures(7),
            LinearRegression()
        ))),
        ('conv2D', image.Convolution(shape=shape, kernel=np.ones((int(0.2 * shape[0]), 1)))),

    ])

    mask, _ = pipe.fit_transform(heatmap)
    pipe.plot()

    # analyse = image.Analyse()
    # analyse.fit(mask)
    #
    # # analyse.plot()
    #
    # threshold = analyse.get_threshold(0.8)
    # print("threshold", threshold)
    # mask = heatmap.copy()
    #
    # mask[mask < threshold] = 0
    # # heatmap[heatmap >= threshold] = 1

    # heatmap = mask * heatmap

    image.Analyse().fit(heatmap).plot()

    print(pipe.statistics())

    print("mean, std: ", heatmap.mean(), heatmap.std(), reduce(lambda x,y:x*y, shape, 1))

    # sns.heatmap(heatmap, cmap='Reds', ax=ax[1])




"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    data, ts = utils.load_pv    (convert_index_to_time=True)
    print("Data len:", len(data) / 288)
    iterations = 2
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
