if __name__ == "__main__": import __config__

from datasets import utils
from SEAPF.Model import Model

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, mean_absolute_error

from experimental.Experimental import Experimental

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""
    Examination model on specified metrics
"""
if __name__ == "__main__":
    data, ts = utils.load_dataset(convert_index_to_time=True)

    experiment = Experimental()

    # register timeseries
    # experiment.register_dataset(data.iloc[0:288*(30+40)]["Production"])
    experiment.register_dataset(data["Production"])

    # register models
    experiment.register_models([
        Model(latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=30,
              y_bins=60, bandwidth=0.4, zeros_filter_modifier=-0.2, density_filter_modifier=-0.5),
        DecisionTreeRegressor(random_state=0)
    ])

    # register metrics and set theirs names (used int returned pandas dataframe)
    experiment.register_metrics([
        (r2_score, "R2"),
        (mean_absolute_error, "MAE")
    ])

    #make experiment
    predictions = experiment.predict(
        forecast_horizon = 288,
        batch = 288,
        window_length = 288*80,
        # early_stop=288*3 # define early stop - before all data are used
    )

    experiment.models[0].plot() # the last model version is saved, so simply plot it

    plotter = experiment.plot()
    plt.show()

