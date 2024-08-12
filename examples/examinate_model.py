if __name__ == "__main__": import __config__

from datasets import utils
from SEAPF.Model import Model
from matplotlib import pyplot as plt
from SEAPF.Plotter import Plotter
from sklearn.tree import DecisionTreeRegressor
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sklearn.metrics import r2_score, mean_absolute_error


import pandas as pd
import numpy as np

def series_to_Xy(data: pd.Series):
    return (data.index.to_numpy().astype(int)/10**9).astype(int).reshape(-1,1), data.to_numpy().reshape(-1,1)


"""
    Examination model on specified metrics
"""
if __name__ == "__main__":
    data, ts = utils.load_dataset(convert_index_to_time=True)


    models = [
        Model(latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=30,
              y_bins=60, bandwidth=0.4, zeros_filter_modifier=1, density_filter_modifier=-0.5),
        DecisionTreeRegressor(random_state=0)
    ]


    # make dataset and y from timeseries
    dataset = data.iloc[0:288*(30+40)]["Production"]
    # dataset = data["Production"]
    X,y = series_to_Xy(dataset)
    # print(X.shape, y.shape)
    # print(np.concatenate((X,y), axis=1))

    forecast_horizon = 288
    batch = 288
    window_length = 288*30 # 120 days to fit model
    fh = [288] # one day ahead
    cv = SlidingWindowSplitter(window_length=window_length, fh=fh)

    # initialize prediction results
    forecast_start_point = forecast_horizon + window_length - 1
    predictions = [[0] * forecast_start_point for _ in models]

    for i, (train, test) in enumerate(cv.split(y)):
        #print(train, test)
        if i % batch == 0:
            print(f"Batch learning. Iter: {i}")
            for model in models:
                model.fit(X=X[train], y=y[train])

        for j, model in enumerate(models):
            prediction = model.predict(X=X[test])
            predictions[j].append(*prediction)
            #print("X[test]", prediction, predictions[i])

    metrics = [
        r2_score, mean_absolute_error
    ]

    metric_results = []

    for metric in metrics:
        for prediction in predictions:
            result = metric(prediction[forecast_start_point:], y[forecast_start_point:, 0])
            metric_results.append(result)

    print(metric_results)

    plotter = Plotter(ts, [dataset.to_numpy(), *predictions], debug=False)
    plotter.show()
    plt.show()
