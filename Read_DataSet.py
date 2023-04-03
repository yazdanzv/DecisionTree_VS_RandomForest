import pandas as pd
import numpy as np

PATH = ".\\covtype.csv"
SPLIT_RATIO = 0.7


def read_x():
    df = pd.read_csv(PATH, usecols=[num for num in range(0, 54)])
    x_data = np.array(df.values.tolist())
    return x_data


def read_y():
    df = pd.read_csv(PATH, usecols=[54])
    y_data = np.array(df.values.tolist())
    return y_data


def split_data(x_data: np.array, y_data: np.array):
    # Shuffle the DataSet
    randomize = np.arange(len(x_data))
    np.random.shuffle(randomize)
    x_data = x_data[randomize]
    y_data = y_data[randomize]
    # Split DataSet
    x_train = x_data[0:int(len(x_data) * SPLIT_RATIO)]
    x_test = x_data[int(len(x_data) * SPLIT_RATIO):]
    y_train = y_data[0:int(len(y_data) * SPLIT_RATIO)]
    y_test = y_data[int(len(y_data) * SPLIT_RATIO):]
    return x_data, x_train, x_test, y_data, y_train, y_test


x_data, x_train, x_test, y_data, y_train, y_test = split_data(read_x(), read_y())
