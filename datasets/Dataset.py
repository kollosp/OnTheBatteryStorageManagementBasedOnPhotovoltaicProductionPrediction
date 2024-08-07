import numpy as np
from typing import List

class Dataset:
    def __init__(self):
        pass

    @staticmethod
    def get(field=1, filename="dataset.xlsx"):
        # File beginning:
        # Timestamp;Production;Elevation;Model;Weather;Cloudiness variability;Overall sunny index
        # 1656536400.000;0.000;0.000;0.000;0.000;0.000;0.000
        # 1656536700.002;0.000;0.000;0.000;1.000;0.000;0.000
        data = np.genfromtxt(filename, delimiter=";")

        ts = data[1:, 0]
        column = data[1:, field]

        return ts, column

    @staticmethod
    def get_full(filename="dataset.csv"):
        return np.genfromtxt(filename, delimiter=";", dtype=None)

    @staticmethod
    def save(data:np.ndarray, headers:List[str], filename="dataset_save.csv"):
        header = ";".join(headers)
        np.savetxt(filename, data, header=header, delimiter=";", fmt="%.3f", comments='')

    @staticmethod
    def timeseries_to_dataset(data: List[np.ndarray], window_size: int, ts=None, skip:int = None):
        """
        That function change list of timeseries (the same length) into an array of windows of specified size.
        A = [a1,a2,..,an] window_size = 2
        :param tss: list of np.ndarray of the same length (may it be also 2D array)
        :param window_size: a number of observation selected as features
        :param ts: timestamps array (optional) if exists is appended as the first column to the dataset
        :param skip: parameter that allows to skip some windows (to adjust data obtained from different window_sizes)
        :return:
            [[a1, a2],
            [a2,a3],
            ...,
            [an-1,an]]

        The returned array represents observation and its features that is consists with the schema that is used in ML.

        """

        end = len(data[0]) - int(window_size)
        # override if passed
        if skip is not None:
            end = len(data[0]) - skip

        dss = [np.array([ts[i: i+window_size] for i in range(0, end)]) for ts in data]
        ret = np.hstack(dss)


        if ts is not None:
            ret = np.hstack([ts[:end].reshape(-1, 1), ret])

        return ret
