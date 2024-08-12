if __name__ == "__main__": import __config__

from datasets import utils
from SEAPF.Model import Model
from matplotlib import pyplot as plt
from SEAPF.Plotter import Plotter

from sklearn.tree import DecisionTreeRegressor

def f():
    data, ts = utils.load_dataset(convert_index_to_time=True)
    # print(data.head(), data.columns)

    # Use all columns to create X
    # X = utils.timeseries_to_dataset([data[i] for i in data.columns], window_size=1)

    production = data["Production"].to_numpy()
    ts = ts.reshape(-1, 1)
    print(ts)
    model = Model(latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=30,
                 y_bins=60, bandwidth=0.4)
    model2 = Model(latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=30,
                 y_bins=60, bandwidth=0.4, interpolation=True)
    model.fit(X=ts, y=production,  zeros_filter_modifier = -0.4, density_filter_modifier = -0.5)
    model2.fit(X=ts, y=production,  zeros_filter_modifier = -0.4, density_filter_modifier = -0.5)

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X=ts, y=production)
    pred1 = regressor.predict(ts)
    pred = model.predict(ts)
    pred3 = model2.predict(ts)
    model.plot()
    plotter = Plotter(ts, [production, pred, pred1, pred3], debug=True)
    plotter.show()
    plt.show()


"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    f()