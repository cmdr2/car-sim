import numpy as np

from .track import TrackSpline


class Car:
    def __init__(self, mass, tire_grip, wheel_base, max_steering_angle):
        self.mass = mass
        self.tire_grip = tire_grip
        self.wheel_base = wheel_base
        self.max_steering_angle = max_steering_angle
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.heading = 0.0  # Car's orientation in radians
        self.steering_angle = 0.0
        self.speed = 0.0

    def apply_controls(self, throttle, brake, steering_input):
        # Steering input is in the range [-1, 1]
        self.steering_angle = steering_input * self.max_steering_angle

        # Apply acceleration
        force = throttle * 500 - brake * 300  # Arbitrary force values
        acceleration = force / self.mass
        self.speed += acceleration

        # Update position
        self.velocity = np.array([self.speed * np.cos(self.heading), self.speed * np.sin(self.heading)])
        self.position += self.velocity

        # Apply Ackermann steering and centripetal force
        self._apply_steering()

    def _apply_steering(self):
        turning_radius = self.wheel_base / np.tan(self.steering_angle)
        angular_velocity = self.speed / turning_radius

        self.heading += angular_velocity

    def get_lateral_force(self):
        # Lateral force using simplified Pacejka formula
        slip_angle = np.arctan2(self.velocity[1], self.velocity[0]) - self.heading
        lateral_force = self.tire_grip * np.sin(slip_angle)
        return lateral_force

    def update_drift(self):
        lateral_force = self.get_lateral_force()
        if np.abs(lateral_force) > self.tire_grip:
            drift_factor = lateral_force / self.tire_grip
            self.velocity = self.velocity * (1 - drift_factor)  # Simulate drift

    def detect_boundaries(self, track_spline: TrackSpline):
        # Check if the car is out of bounds using the track spline
        closest_point = track_spline.get_closest_point(self.position)
        distance_to_track = np.linalg.norm(self.position - closest_point)
        if distance_to_track > track_spline.track_width:
            self.speed = 0  # Stop the car if it's off-track
            return True
        return False
