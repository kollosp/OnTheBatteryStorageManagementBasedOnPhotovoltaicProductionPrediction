from sklearn.neighbors import KernelDensity
import numpy as np

from lib.Plotter import Plotter


class ApplyKde:
    def __init__(self, kernel, bandwidth, bins):
        self._kernel = kernel
        self._bandwidth = bandwidth
        self._bins = bins

    def __call__(self, a):
        b = a[~np.isnan(a)].reshape(-1, 1)

        kde = KernelDensity(kernel=self._kernel, bandwidth=self._bandwidth)
        kde = kde.fit(b.reshape(-1, 1))
        log_dens = kde.score_samples(self._bins)

        return np.exp(log_dens)


class Overlay:
    def __init__(self, overlay, _y_bins, _bandwidth ):
        self._overlay = overlay
        self._y_bins = _y_bins
        self._bandwidth = _bandwidth

        self._heatmap = np.apply_along_axis(lambda a: np.histogram(a[~np.isnan(a)], bins=self._y_bins)[0], 0,
                                            self._overlay)
        self._heatmap = np.apply_along_axis(lambda a: (100 * a / np.nansum(a)).astype(int), 0, self._heatmap)
        self._max_value_in_overlay = overlay[~np.isnan(overlay)].max()
        r = (0, self._max_value_in_overlay)
        bins_no = self._y_bins
        self._bins = np.array([r[0] + (r[1] - r[0]) * i / (bins_no - 1) for i in range(bins_no)]).reshape(-1, 1)  # bins len

        apply_kde = ApplyKde(kernel="gaussian", bandwidth=self._bandwidth, bins=self._bins)
        self._kde = np.apply_along_axis(apply_kde, 0, self._overlay)

    @property
    def bins(self):
        return self._bins

    @property
    def overlay(self):
        return self._overlay

    @property
    def kde(self):
        return self._kde

    @property
    def heatmap(self):
        return self._heatmap

    def plot(self):
        fig, ax, cols, rows = Plotter.plot_2D_histograms(self.heatmap, self.kde)
        ts = list(range(self._kde.shape[0]))
        for i in range(self._kde.shape[1]):
            y = i // cols
            x = i % rows
            axis = ax[y,x]

            threshold = self.density_based_filter_threshold(i)
            axis.plot(ts, [threshold] * len(ts), color="r")
            threshold = self.zeros_filter_threshold(i) * self._kde.shape[1] / self._max_value_in_overlay
            axis.axvline(x=threshold, color="r")

    def zeros_filter_threshold(self,i):
        d = self._kde[:, i]
        threshold = sum(d * [self._max_value_in_overlay * i / len(d) for _, i in enumerate(d)]) / d.sum()
        return threshold

    def apply_zeros_filter(self):
        for i in range(self._overlay.shape[1]):
            self._overlay[:, i] = Overlay.highpass_filter(self._overlay[:, i], self.zeros_filter_threshold(i))

        return Overlay(self._overlay, self._y_bins, self._bandwidth)

    def density_based_filter_threshold(self, i):
        return self._kde[:, i].mean()

    def apply_density_based_filter(self):
        for i in range(self._overlay.shape[1]):
            d = self._kde[:, i]
            passing_bools = d > self.density_based_filter_threshold(i)
            bins_boundaries = np.array([[self._max_value_in_overlay * i / len(d), self._max_value_in_overlay * (i+1) / len(d)]  for i,_ in enumerate(d)])
            bins_boundaries = bins_boundaries[passing_bools]
            for j in range(self._overlay.shape[0]):
                exists_in = False
                for _,boundaries in enumerate(bins_boundaries):
                    if boundaries[0] <= self._overlay[j, i] < boundaries[1]:
                        exists_in = True

                if not exists_in:
                    self._overlay[j, i] = np.nan

        return Overlay(self._overlay, self._y_bins, self._bandwidth)


    @staticmethod
    def highpass_filter(data: np.ndarray, threshold: float):
        """
            The function apply high pass filter below the given threshold. It removes all values from overlay those not
            excesses the threshold
        """
        data[data < threshold] = np.nan
        return data

    @staticmethod
    def lowpass_filter(data: np.ndarray, threshold: float):
        """
            The function apply low pass filter above the given threshold. It removes all values from overlay those
            excesses the threshold
        """
        data[data > threshold] = np.nan
        return data
