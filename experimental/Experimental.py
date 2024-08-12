import pandas as pd
from .utils import series_to_Xy
from typing import List
from sktime.forecasting.model_selection import SlidingWindowSplitter


from SEAPF.Plotter import Plotter

class Metric:
    def __init__(self, metric, name):
        self._metric = metric
        self._name = name
    def __call__(self, *args, **kwargs):
        return self._metric(*args, **kwargs)
    def __str__(self):
        return self._name


class Experimental:
    def __init__(self):
        self._predictions = None
        self._dataset = None
        self._X, self._y = None, None
        self._models = []
        self._metrics = []

    @property
    def predictions(self):
        return self._dataset

    @property
    def dataset(self):
        return self._dataset

    @property
    def models(self):
        return self._models

    @property
    def metrics(self):
        return self._metrics

    def register_dataset(self, dataset: pd.DataFrame):
        self._dataset = dataset
        self._X, self._y = series_to_Xy(self._dataset)
        print(f"Dataset Registered. Shape: X={self._X.shape}, y={self._y.shape}")

    def register_model(self, model):
        self._models.append(model)
    def register_models(self, models : List):
        for model in models:
            self._models.append(model)

    def register_metric(self, metric, name):
        self._metrics.append(Metric(metric, name))

    def register_metrics(self, metrics: List):
        """
        Register multiple metrics
        :param metrics: a list of tuples containing [(callable, name), (callable, name), ...]
        """
        for metric in metrics:
            self._metrics.append(Metric(metric[0], metric[1]))

    def predict(self, forecast_horizon = 288, batch = 288, window_length = 288 * 30, early_stop=None):
        X, y = self._X, self._y

        fh = [forecast_horizon]
        cv = SlidingWindowSplitter(window_length=window_length, fh=fh)

        # initialize prediction results
        forecast_start_point = forecast_horizon + window_length - 1
        predictions = [[0] * len(y) for _ in self._models]

        splits = cv.get_n_splits(y)

        for i, (train, test) in enumerate(cv.split(y)):
            # print(train, test)
            if i % batch == 0:
                print(f"Batch learning. Iter: {i} ~ {int(100 * i/splits)}%")
                for model in self._models:
                    model.fit(X=X[train], y=y[train])

            for j, model in enumerate(self._models):
                prediction = model.predict(X=X[test])
                s = forecast_start_point + i
                predictions[j][s:s + len(prediction)] = prediction
                # print(predictions[j][0:])
                # print("X[test]", prediction, predictions[i])

            if early_stop is not None and i > early_stop:
                print(f"Early stop on i={i} > {early_stop}")
                break


        metrics_results = {}
        for metric in self._metrics:
            metrics_results[f"{metric}"] = []
            for model, prediction in zip(self._models, predictions):
                result = metric(prediction[forecast_start_point:], y[forecast_start_point:])
                metrics_results[f"{metric}"].append(result)

        metrics_results["Models"] = [str(m) for m in self._models]
        metrics_results = pd.DataFrame(metrics_results)
        metrics_results.set_index("Models", inplace=True)
        print(metrics_results)

        self._predictions = pd.DataFrame.from_dict({str(model): prediction for model, prediction in zip(self._models, predictions)})
        return self._predictions

    def statistics(self):
        pass

    def plot(self):
        plotter = Plotter(self._X[:, 0], [self._dataset.to_numpy(),
                                                *[self._predictions[column].to_numpy() for column in self._predictions.columns]], debug=False)
        return plotter.show()

