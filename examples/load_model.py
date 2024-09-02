if __name__ == "__main__": import __config__
import pandas as pd
from datasets import utils
from sktimeSEAPF.Model import Model
from matplotlib import pyplot as plt
from utils.Plotter import Plotter

from sklearn.tree import DecisionTreeRegressor

def f():
    data, ts = utils.load_dataset(convert_index_to_time=True)
    # print(data.head(), data.columns)

    # Use all columns to create X
    # X = utils.timeseries_to_dataset([data[i] for i in data.columns], window_size=1)

    train_test_split = 288*80
    test_len = 288*30

    y_train, y_test = data["Production"][:train_test_split], data["Production"][train_test_split:train_test_split+test_len]

    model = Model(latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=90,
                 y_bins=90, bandwidth=0.4, zeros_filter_modifier = -0.4, density_filter_modifier = -0.5)
    model.fit(y=y_train)
    fh = [i-train_test_split for i in range(train_test_split, train_test_split+test_len)]

    pred = model.predict(fh=fh)
    model.plot()


    print(y_train.index)
    # print_data = pd.DataFrame({'s1': data["Production"], 's2': pred})
    # print_data.plot()
    plotter = Plotter(pred.index, [pred.values], debug=True)
    plotter.show()
    plt.show()


"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    f()
