import numpy as np


def interpolate_curve(x, points):
    x_points, y_points = zip(*points)
    return np.interp(x, x_points, as_np_array(y_points))


def weighted_sum(values, weights, print_labels=False, labels=None):
    values = np.array(values)
    weights = np.array(weights).reshape(-1, 1)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if print_labels:
        x = values * weights / np.sum(weights)
        print("")
        for l, v in zip(labels, x):
            print(l, v)
        print("")

    return np.sum(values * weights, axis=0) / np.sum(weights)


def as_np_array(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr
