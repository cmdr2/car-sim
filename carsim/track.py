import numpy as np
from scipy.interpolate import CubicSpline


class TrackSpline:
    def __init__(self, control_points, track_width):
        self.track_width = track_width
        self.spline_x = CubicSpline(np.arange(len(control_points)), [p[0] for p in control_points])
        self.spline_y = CubicSpline(np.arange(len(control_points)), [p[1] for p in control_points])

    def get_point(self, t):
        return np.array([self.spline_x(t), self.spline_y(t)])

    def get_closest_point(self, position):
        # Naive implementation to find the closest point on the spline to the given position
        min_dist = float("inf")
        closest_point = None
        for t in np.linspace(0, len(self.spline_x.c) - 1, 100):  # Sampling 100 points on the spline
            point = self.get_point(t)
            dist = np.linalg.norm(position - point)
            if dist < min_dist:
                min_dist = dist
                closest_point = point
        return closest_point
