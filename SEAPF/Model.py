from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
from Solar import Solar
from matplotlib import pyplot as plt
from datetime import datetime as dt
from typing import List
# package imports
from .Optimized import Optimized
from .Plotter import Plotter
from .Overlay import Overlay

class Model:
    def __init__(self,
                 latitude_degrees: float,
                 longitude_degrees: float,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 bandwidth: float = 0.2,
                 window_size: int = None,
                 enable_debug_params: bool = False,
                 zeros_filter_modifier:float=0,
                 density_filter_modifier:float=0):
        self._zeros_filter_modifier = zeros_filter_modifier
        self._density_filter_modifier = density_filter_modifier
        self._x_bins = x_bins
        self._bandwidth = bandwidth
        self._y_bins = y_bins
        self._latitude_degrees = latitude_degrees
        self._longitude_degrees = longitude_degrees
        self._model_representation = None
        self._elevation_bins = None
        self._overlay = None
        self._heatmap = None
        self._kde = None
        self._enable_debug_params = enable_debug_params
        self._ws = window_size  # if set then fit function performs moving avreage on the input data

    def fit(self, X: np.ndarray, y: np.ndarray, zeros_filter_modifier:float | None = None, density_filter_modifier:float | None = None):
        """
        Fit function that is similar to sklearn scheme X contains features while y contains corresponding correct values
        :param X: it should be 2D array [[ts1],[ts2],[ts3],[ts4],...] containing timestamps
        :param y: it should be 2D array [[y1],[y2],[y3],[y4],...] containing observations made at the corresponding timestamps
        :return: self
        """

        # model is prepared to work with only one param in X
        ts = X[:, 0] #.reshape(1,-1).squeeze() # reshape to (n, 1) and remove last axis
        data = y[:, 0] #.reshape(1,-1).squeeze()  # reshape to (n, 1) and remove last axis

        if self._ws is not None:
            data = Optimized.window_moving_avg(data, window_size=self._ws, roll=True)
        # calculate elevation angles for the given timestamps
        elevation = Solar.elevation(Optimized.from_timestamps(ts), self._latitude_degrees,
                                        self._longitude_degrees) * 180 / np.pi

        # remove negative timestamps
        elevation[elevation <= 0] = 0
        # create assignment series, which will be used in heatmap processing
        days_assignment = Optimized.date_day_bins(ts)
        elevation_assignment, self._elevation_bins = Optimized.digitize(elevation, self._x_bins)
        overlay = Optimized.overlay(data, elevation_assignment, days_assignment)

        self._overlay = Overlay(overlay, self._y_bins, self._bandwidth)

        if zeros_filter_modifier is None:
            zeros_filter_modifier = self._zeros_filter_modifier
        if density_filter_modifier is None:
            density_filter_modifier = self._density_filter_modifier

        self._overlay = self._overlay.apply_zeros_filter(modifier=zeros_filter_modifier)\
            .apply_density_based_filter(modifier=density_filter_modifier)
        self._model_representation = np.apply_along_axis(lambda a: self._overlay.bins[np.argmax(a)], 0, self._overlay.kde).flatten()

        return self

    def plot(self):
        fig, ax = plt.subplots(3)
        ov = self._overlay.overlay
        Plotter.plot_overlay(ov, fig=fig, ax=ax[0])
        x = list(range(ov.shape[1]))
        ax[0].plot(x, self._model_representation, color="r")

        # compute mean values
        # mean = np.apply_along_axis(lambda a: np.nanmean(), 0, self._overlay)
        mean = np.nanmean(ov, axis=0)
        mx = np.nanmax(ov, axis=0)
        mi = np.nanmin(ov, axis=0)
        ax[0].plot(x, mean, color="orange")
        ax[0].plot(x, mx, color="orange")
        ax[0].plot(x, mi, color="orange")

        ax[1].imshow(self._overlay.heatmap, cmap='Blues', origin='lower')
        ax[2].imshow(self._overlay.kde, cmap='Blues', origin='lower')

        # Plotter.plot_2D_histograms(self._overlay.heatmap, self._overlay.kde)
        self._overlay.plot()
        return fig, ax

    def predict(self, X: np.ndarray):

        ts = X[:, 0] #.reshape(1, -1).squeeze()  # reshape to (n, 1) and remove last axis
        if self._model_representation is None:
            raise RuntimeError("Model.predict: Use fit method first!")

        elevation = Solar.elevation(Optimized.from_timestamps(ts), self._latitude_degrees,
                                        self._longitude_degrees) * 180 / np.pi

        return Optimized.model_assign(self._model_representation, self._elevation_bins, elevation, self._enable_debug_params)

    def set_step_ahead_forecasting(self):
        pass

    def __str__(self):
        # return "Model representation: " + str(self._model_representation) + \
        #     " len(" + str(len(self._model_representation)) + ")" + \
        #     "\nBins: " + str(self._elevation_bins) + " len(" + str(len(self._elevation_bins)) + ")"
        return "SEAPF"