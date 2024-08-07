from datasets import utils
from lib.Model import Model
from matplotlib import pyplot as plt



"""
    Show dataset if the script was run directly instead of being loaded as package
"""
if __name__ == "__main__":
    data, ts = utils.load_dataset(convert_index_to_time=True)
    # print(data.head(), data.columns)

    # Use all columns to create X
    # X = utils.timeseries_to_dataset([data[i] for i in data.columns], window_size=1)

    production = data["Production"].to_numpy()
    print(production)

    model = Model(latitude_degrees=utils.LATITUDE_DEGREES, longitude_degrees=utils.LONGITUDE_DEGREES, x_bins=30,
                 y_bins=60, bandwidth=0.4)
    model.fit(ts=ts, data=production)

    fig, ax = model.plot()
    fig.show()
    plt.show()