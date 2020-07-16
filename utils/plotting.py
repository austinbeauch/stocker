import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_ticker(input_prices, predicted, truth, ticker=None, **kwargs):
    if input_prices.ndim > 1:
        c = 0
        for inpt, pred, true, tick in zip(input_prices, predicted, truth, ticker):
            plot_prediction(inpt, pred, true, ticker=tick, counter=c, **kwargs)
            c+=1
    else:
        plot_prediction(input_prices, predicted, truth, ticker=ticker, **kwargs)


def plot_prediction(x, y, pred, scaler=None, ticker=None, batch=None, counter=None):
    test_true_data = np.empty((len(x) + len(y), 3,))
    test_true_data[:] = np.nan
    test_true_data[:len(x), 0] = x.detach().numpy()
    test_true_data[len(x):, 1] = y.detach().numpy()
    test_true_data[len(x):, 2] = pred.detach().numpy()

    if scaler is not None:
        test_true_data = scaler.inverse_transform(test_true_data)

    test_true_data = pd.DataFrame(test_true_data)

    plt.figure(figsize=(12, 6))
    plt.plot(test_true_data.index, test_true_data[0], label="Input")
    plt.plot(test_true_data.index, test_true_data[1], label="True output")
    plt.plot(test_true_data.index, test_true_data[2], label="Predicted output")
    plt.legend()
    if ticker is not None:
        plt.title(ticker)
    # plt.ylim(-2, 2)
    plt.xlabel("Days")
    plt.ylabel("Price")
    # plt.show(block=False)
    plt.savefig(f"./images/{batch}_{counter}")
    plt.close()
