import numpy as np


WHEELSPIN_FACTOR = 0.01
WHEELSPIN_TORQUE_SENSITIVITY = 2
LATERAL_FRICTION_MULTIPLIER = 2
VERTICAL_LOAD_TO_GRIP_FACTOR = 0.6
TIRE_DEFORMATION_FACTOR = 0.1


def get_max_allowed_traction_force_magnitude(effective_grip, vertical_load):
    """
    Calculate the max traction force magnitude on the wheel, beyond which it starts slipping.

    :param effective_grip: Friction-like coefficient for the tire.
    :param vertical_load: Vertical force vector applied on the wheel (newtons).

    :return: Max traction force magnitude (newtons).
    """
    vertical_load_magnitude = np.linalg.norm(vertical_load)

    return effective_grip * np.power(vertical_load_magnitude, VERTICAL_LOAD_TO_GRIP_FACTOR)


def get_reaction_force_on_axle(
    engine_torque,
    tire_radius,
    max_traction_force,
    tire_stiffness,
    inertial_force,
    tire_forward_direction,
    tire_up_direction,
):

    deformation_absorption = 1 - TIRE_DEFORMATION_FACTOR * tire_radius / tire_stiffness

    # Compute engine force with deformation, in the negative tire forward direction
    engine_force = -tire_forward_direction * (engine_torque / tire_radius) * deformation_absorption

    # Compute tire right vector as the cross product of tire forward and tire up vectors
    tire_right_direction = np.cross(tire_forward_direction, tire_up_direction)

    # Project the inertial force onto the tire right direction
    lateral_inertial_force = (
        np.dot(inertial_force, tire_right_direction) * tire_right_direction * deformation_absorption
    )

    # Normalize lateral inertial force by friction multiplier
    normalized_lateral_inertial_force = lateral_inertial_force / LATERAL_FRICTION_MULTIPLIER

    # Sum the engine force and the lateral inertial force
    total_force = engine_force + normalized_lateral_inertial_force

    # Use magnitudes for comparison
    total_force_magnitude = np.linalg.norm(total_force)
    max_traction_force_magnitude = np.linalg.norm(max_traction_force)

    if total_force_magnitude <= max_traction_force_magnitude:
        # No slip, return reaction force in tire forward direction with deformation absorption
        reaction_force = tire_forward_direction * np.linalg.norm(engine_force) * deformation_absorption
        return reaction_force

    # Slipping case
    d = total_force_magnitude - max_traction_force_magnitude
    reaction_torque = engine_torque * (1 - WHEELSPIN_FACTOR * d**WHEELSPIN_TORQUE_SENSITIVITY)
    reaction_force = tire_forward_direction * (reaction_torque / tire_radius) * deformation_absorption

    return reaction_force
