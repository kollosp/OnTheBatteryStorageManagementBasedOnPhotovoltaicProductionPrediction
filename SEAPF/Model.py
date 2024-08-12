from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.metrics import r2_score
from Solar import Solar
from matplotlib import pyplot as plt
from datetime import datetime as dt
from typing import List
# package imports
from .Optimized import Optimized
from .Plotter import Plotter
from .Overlay import Overlay

class Model(RegressorMixin, BaseEstimator):
    def __init__(self,
                 latitude_degrees: float = 51,
                 longitude_degrees: float = 14,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 bandwidth: float = 0.2,
                 window_size: int = None,
                 enable_debug_params: bool = False,
                 zeros_filter_modifier:float=0,
                 density_filter_modifier:float=0,
                 interpolation=False):
        self.zeros_filter_modifier = zeros_filter_modifier
        self.density_filter_modifier = density_filter_modifier
        self.x_bins = x_bins
        self.bandwidth = bandwidth
        self.y_bins = y_bins
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.interpolation = interpolation
        # self.model_representation_ = None
        # self.elevation_bins_ = None
        # self.overlay_ = None
        # self.heatmap_ = None
        # self.kde_ = None
        self.enable_debug_params = enable_debug_params
        self.window_size = window_size  # if set then fit function performs moving avreage on the input data

    def fit(self, X: np.ndarray, y: np.ndarray, zeros_filter_modifier:float | None = None, density_filter_modifier:float | None = None):
        """
        Fit function that is similar to sklearn scheme X contains features while y contains corresponding correct values
        :param X: it should be 2D array [[ts1],[ts2],[ts3],[ts4],...] containing timestamps
        :param y: it should be 2D array [[y1],[y2],[y3],[y4],...] containing observations made at the corresponding timestamps
        :param zeros_filter_modifier:
        :param density_filter_modifier:
        :return: self
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # model is prepared to work with only one param in X
        ts = X[:, 0] #.reshape(1,-1).squeeze() # reshape to (n, 1) and remove last axis
        data = y #y[:, 0] #.reshape(1,-1).squeeze()  # reshape to (n, 1) and remove last axis

        if self.window_size is not None:
            data = Optimized.window_moving_avg(data, window_size=self.window_size, roll=True)
        # calculate elevation angles for the given timestamps
        elevation = Solar.elevation(Optimized.from_timestamps(ts), self.latitude_degrees,
                                        self.longitude_degrees) * 180 / np.pi

        # remove negative timestamps
        elevation[elevation <= 0] = 0
        # create assignment series, which will be used in heatmap processing
        days_assignment = Optimized.date_day_bins(ts)
        elevation_assignment, self.elevation_bins_ = Optimized.digitize(elevation, self.x_bins)
        overlay = Optimized.overlay(data, elevation_assignment, days_assignment)

        self.overlay_ = Overlay(overlay, self.y_bins, self.bandwidth)

        if zeros_filter_modifier is None:
            zeros_filter_modifier = self.zeros_filter_modifier
        if density_filter_modifier is None:
            density_filter_modifier = self.density_filter_modifier

        self.overlay_ = self.overlay_.apply_zeros_filter(modifier=zeros_filter_modifier)\
            .apply_density_based_filter(modifier=density_filter_modifier)
        self.model_representation_ = np.apply_along_axis(lambda a: self.overlay_.bins[np.argmax(a)], 0, self.overlay_.kde).flatten()

        return self

    def plot(self):
        fig, ax = plt.subplots(3)
        ov = self.overlay_.overlay
        Plotter.plot_overlay(ov, fig=fig, ax=ax[0])
        x = list(range(ov.shape[1]))
        ax[0].plot(x, self.model_representation_, color="r")

        # compute mean values
        # mean = np.apply_along_axis(lambda a: np.nanmean(), 0, self.overlay_)
        mean = np.nanmean(ov, axis=0)
        mx = np.nanmax(ov, axis=0)
        mi = np.nanmin(ov, axis=0)
        ax[0].plot(x, mean, color="orange")
        ax[0].plot(x, mx, color="orange")
        ax[0].plot(x, mi, color="orange")

        ax[1].imshow(self.overlay_.heatmap, cmap='Blues', origin='lower')
        ax[2].imshow(self.overlay_.kde, cmap='Blues', origin='lower')

        # Plotter.plot_2D_histograms(self.overlay_.heatmap, self.overlay_.kde)
        self.overlay_.plot()
        return fig, ax

    def predict(self, X: np.ndarray, y=None):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        ts = X[:, 0] #.reshape(1, -1).squeeze()  # reshape to (n, 1) and remove last axis
        if self.model_representation_ is None:
            raise RuntimeError("Model.predict: Use fit method first!")

        elevation = Solar.elevation(Optimized.from_timestamps(ts), self.latitude_degrees,
                                        self.longitude_degrees) * 180 / np.pi

        return Optimized.model_assign(self.model_representation_,
                                      self.elevation_bins_,
                                      elevation,
                                      self.enable_debug_params,
                                      interpolation=self.interpolation)

    def set_step_ahead_forecasting(self):
        pass

    def score(self, X, y):
        # poor values - function crated only for api consistence
        #
        # X, y = check_X_y(X, y)
        # pred = self.predict(X,y)
        # return r2_score(pred, y)
        return 0.6


    def __str__(self):
        # return "Model representation: " + str(self.model_representation_) + \
        #     " len(" + str(len(self.model_representation_)) + ")" + \
        #     "\nBins: " + str(self.elevation_bins_) + " len(" + str(len(self.elevation_bins_)) + ")"
        return "SEAPF (" + str(self.get_params()) + ")"