import numpy as np
from scipy.interpolate import interp1d


def interpolate_curve(points):
    x_points, y_points = zip(*points)
    interpolation_function = interp1d(x_points, y_points, kind="linear", fill_value="extrapolate")
    return lambda x: interpolation_function(x)


def weighted_sum(values, weights, print_labels=False, labels=None):
    values = np.array(values)
    weights = np.array(weights)

    if print_labels:
        x = values * weights / np.sum(weights)
        print("")
        for l, v in zip(labels, x):
            print(f"{l}: {v:0.2f}")
        print("")

    return np.sum(values * weights) / np.sum(weights)
