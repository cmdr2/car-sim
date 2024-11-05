"""
Calculates the "friction-like" multiplier for a given tire (or an array of tires). This module takes into account the
tire and road conditions, and produces a factor that can be multiplied by the vertical load (to get the grip force).

Use the `get_tire_grip` function to calculate this factor.

# Example usage (single tire)
tire_material_coeff = 1.0
tread_amount = 0  # Amount of tread on the tire. 0 is no tread (i.e. full slicks), and increasing tread towards 1
road_type = "asphalt"
road_condition = 0.8  # 0 is slippery, 1 is ideal
tire_width = 305  #  mm
tire_hardness_factor = 0.5
tire_pressure = 30  # PSI
tire_temperature = 90  # C
tire_wear = 0.5
camber = -2.5  # degrees

effective_grip = get_tire_grip(
    tire_material_coeff,
    tread_amount,
    road_type,
    road_condition,
    tire_width,
    tire_hardness_factor,
    tire_pressure,
    tire_temperature,
    tire_wear,
    camber,
)
print(f"Effective Grip: {effective_grip}")


# Example vectorized usage (multiple tires in different conditions)
tire_material_coeff = [1.0, 1.0]
tread_amount = [0, 0]
road_type = ["asphalt", "gravel"]
road_condition = [0.8, 0.4]
tire_width = [305, 305]  #  mm
tire_hardness_factor = [0.5, 0.5]
tire_pressure = [30, 30]  # PSI
tire_temperature = [100, 90]  # C
tire_wear = [0.2, 0.5]
camber = [-2.5, -2.5]  # degrees

effective_grip = get_tire_grip(
    tire_material_coeff,
    tread_amount,
    road_type,
    road_condition,
    tire_width,
    tire_hardness_factor,
    tire_pressure,
    tire_temperature,
    tire_wear,
    camber,
)
print(f"Effective Grip: {effective_grip}")

"""

import numpy as np

from carsim.util import interpolate_curve, weighted_sum, as_np_array

OPTIMAL_CAMBER_FOR_REFERENCE_TIRE = -3.5  # Base optimal camber for reference width tires
REFERENCE_TIRE_WIDTH = 305  # Reference width in mm (standard tire width)
REFERENCE_OPTIMAL_PRESSURE = 28  # psi
REFERENCE_HARDNESS_TO_TEMP_LOW_CURVE = [
    (0, 45),
    (0.4, 55),
    (0.5, 75),
    (0.6, 80),
    (0.7, 80),
    (0.8, 85),
    (0.9, 85),
    (1.0, 85),
]  # https://web.archive.org/web/20240714102648/https://simracingsetup.com/f1-24/f1-24-ideal-tyre-temperatures/
REFERENCE_HARDNESS_TO_TEMP_HIGH_CURVE = [
    (0, 65),
    (0.4, 75),
    (0.5, 105),
    (0.6, 105),
    (0.7, 110),
    (0.8, 115),
    (0.9, 120),
    (1.0, 120),
]
REFERENCE_TIRE_WEAR_TO_GRIP_CURVE = [
    (0.0, 1.0),
    (0.1, 0.95),
    (0.2, 0.9),
    (0.3, 0.82),
    (0.4, 0.75),
    (0.5, 0.6),
    (0.6, 0.5),
    (0.7, 0.2),
    (0.8, 0.08),
    (0.9, 0.03),
    (1.0, 0),
]
FRICTION_ASPHALT = 1.0
FRICTION_CONCRETE = 1.1
FRICTION_DIRT = 0.7
FRICTION_GRAVEL = 0.6
FRICTION_GRASS = 0.5
FRICTION_ICE = 0.1

# Weightage of factors
SURFACE_FRICTION_EFFECT = 0.5
TIRE_HARDNESS_EFFECT = 1
TIRE_PRESSURE_EFFECT = 0.3
TIRE_WIDTH_EFFECT = 0.2
CAMBER_EFFECT = 0.2
TEMPERATURE_EFFECT = 0.5
TIRE_WEAR_EFFECT = 1


# Friction based on material properties of tire and road
def material_friction(tire_material_coefficient, tread_amount, road_type, road_condition):
    """
    Calculate the base static friction coefficient based on tire and road material.

    :param tire_material_coefficient: Static friction coefficient for tire material (μ_s).
    :param tread_amount: Factor (0 to 1) representing the amount of tread. 0 is no tread (i.e. slicks).
    :param road_type: Type of surface. Accepted values: 'asphalt', 'concrete', 'dirt', 'gravel', 'ice'.
    :param road_condition: Factor (between 0 and 1) representing the road surface conditions.

    :return: Base static friction coefficient.
    """

    # Base friction coefficient for road types in perfect conditions

    road_friction = np.zeros_like(tire_material_coefficient)

    asphalt_mask = road_type == "asphalt"
    concrete_mask = road_type == "concrete"
    dirt_mask = road_type == "dirt"
    gravel_mask = road_type == "gravel"
    grass_mask = road_type == "grass"
    ice_mask = road_type == "ice"

    road_friction[asphalt_mask] = FRICTION_ASPHALT
    road_friction[concrete_mask] = FRICTION_CONCRETE
    road_friction[dirt_mask] = FRICTION_DIRT
    road_friction[gravel_mask] = FRICTION_GRAVEL
    road_friction[grass_mask] = FRICTION_GRASS
    road_friction[ice_mask] = FRICTION_ICE

    road_friction *= np.power(road_condition, 0.3)

    # Adjust tread effectiveness based on road type and condition
    ## asphalt and concrete
    x = road_condition - 0.5
    asphalt_concrete_tread_effect = 1 - 1 * np.power(tread_amount, 0.5) * np.sign(x) * np.power(abs(x), 0.5)

    ## dirt, gravel, grass
    dirt_gravel_grass_tread_effect = 1 + 0.5 * tread_amount

    ## ice
    ice_tread_effect = 1

    tread_effect = (
        (asphalt_mask | concrete_mask) * asphalt_concrete_tread_effect
        + (dirt_mask | gravel_mask | grass_mask) * dirt_gravel_grass_tread_effect
        + ice_mask * ice_tread_effect
    )

    # Calculate final friction coefficient
    friction_coefficient = tire_material_coefficient * road_friction * tread_effect
    return friction_coefficient


def tire_hardness_effect(tire_hardness_factor):
    """
    Calculate the effective friction based on tire hardness.

    :param tire_hardness_factor: Tire hardness (0 to 1, where 0 is soft and 1 is hard).

    :return: Effective friction multiplier based on tire hardness.
    """
    # Tire hardness effect: Soft tires have more grip
    hardness_effect = np.sqrt(1 - tire_hardness_factor)  # Soft tires increase friction

    return hardness_effect


def tire_pressure_effect(tire_pressure):
    """
    Calculate the effective friction based on tire pressure.

    :param tire_pressure: Tire pressure in psi (pounds per square inch).

    :return: Effective friction multiplier based on tire pressure.
    """
    # Tire pressure effect: Decrease friction as pressure increases
    pressure_effect = np.power(REFERENCE_OPTIMAL_PRESSURE / tire_pressure, 0.7)

    return pressure_effect


def tire_width_effect(tire_width):
    """
    Calculate the effective friction based on tire width.

    :param tire_width: Width of the tire in millimeters.

    :return: Effective friction multiplier based on tire width.
    """
    # Tire width effect: Wider tires increase contact area but with diminishing returns
    width_effect = 1 + (tire_width - REFERENCE_TIRE_WIDTH) / (2 * REFERENCE_TIRE_WIDTH)

    return width_effect


# Factor in camber
def camber_effect(camber, tire_width):
    """
    Adjust friction based on camber angle, vertical load, suspension stiffness, and tire width.

    :param camber: Camber angle in degrees.
    :param tire_width: Width of the tire in millimeters.

    :return: Friction adjustment based on camber and tire width.
    """
    # Base optimal camber values based on tire width
    optimal_camber_adjustment = 0.05 * (tire_width - REFERENCE_TIRE_WIDTH) / REFERENCE_TIRE_WIDTH

    # Adjusted optimal camber based on width
    optimal_camber = OPTIMAL_CAMBER_FOR_REFERENCE_TIRE + optimal_camber_adjustment

    # Calculate camber effect based on difference from the adjusted optimal camber
    return np.exp(-np.abs(camber - optimal_camber) / 5)  # Exponential decay for non-optimal camber


def temperature_effect(tire_temperature, tire_hardness_factor):
    """
    Adjust friction based on temperature.

    :param tire_temperature (float): The current tire temperature in degrees Celsius.
    :param tire_hardness_factor (float): The tire hardness value.

    :return: float: Friction adjustment based on temperature.
    """
    low_optimal = interpolate_curve(tire_hardness_factor, REFERENCE_HARDNESS_TO_TEMP_LOW_CURVE)
    high_optimal = interpolate_curve(tire_hardness_factor, REFERENCE_HARDNESS_TO_TEMP_HIGH_CURVE)

    # mask for different temp conditions
    cold_temp_mask = tire_temperature < low_optimal
    optimal_temp_mask = (tire_temperature >= low_optimal) & (tire_temperature <= high_optimal)
    overheated_temp_mask = tire_temperature > high_optimal

    # grips for different temp conditions
    cold_temp_grip = np.power(tire_temperature / low_optimal, 3)  # Decrease grip with temperature

    overheating_factor = (tire_temperature - high_optimal) / (high_optimal - low_optimal)
    overheated_temp_grip = np.clip(1 - overheating_factor, 0, 1)  # Decrease grip, but not below 0
    overheated_temp_grip = np.power(overheated_temp_grip, 3)

    # apply the correct grips
    grip = (
        cold_temp_mask * cold_temp_grip
        + optimal_temp_mask * np.ones_like(temperature_effect)
        + overheated_temp_mask * overheated_temp_grip
    )

    return grip


def tire_wear_effect(tire_wear):
    """
    Calculate the friction effect based on tire wear.

    :param tire_wear: Tire wear as a value between 0 and 1 (0 = new, 1 = worn out).

    :return: Friction adjustment based on tire wear.
    """
    tire_wear = np.clip(tire_wear, 0, 1)

    # Friction effect scales with tire wear
    return interpolate_curve(tire_wear, REFERENCE_TIRE_WEAR_TO_GRIP_CURVE)


# Final function to combine all factors and return the effective friction coefficient (aka "grip")
def get_tire_grip(
    tire_material_coeff,
    tread_amount,
    road_type,
    road_condition,
    tire_width,
    tire_hardness_factor,
    tire_pressure,
    tire_temperature,
    tire_wear,
    camber,
):
    """
    Combine all factors to calculate the final effective friction coefficient.

    :param tire_material_coeff: Static friction coefficient for tire material (μ_s).
    :param tread_amount: Factor (0 to 1) representing the amount of tread. 0 is no tread (i.e. slicks).
    :param road_type: Type of surface. Accepted values: 'asphalt', 'concrete', 'dirt', 'gravel', 'ice'.
    :param road_condition: Factor (0 to 1) representing the road surface conditions.
    :param tire_width: Width of the tire in millimeters.
    :param tire_hardness_factor: Tire hardness (0 to 1, where 0 is soft and 1 is hard).
    :param tire_pressure: Tire pressure in psi (pounds per square inch).
    :param tire_temperature (float): The current tire temperature in degrees Celsius.
    :param tire_wear: Tire wear as a value between 0 and 1 (0 = new, 1 = worn out).
    :param camber: Camber angle in degrees.

    :return: Effective grip
    """

    # ensure these are numpy arrays
    tire_material_coeff = as_np_array(tire_material_coeff)
    tread_amount = as_np_array(tread_amount)
    road_type = as_np_array(road_type)
    road_condition = as_np_array(road_condition)
    tire_width = as_np_array(tire_width)
    tire_hardness_factor = as_np_array(tire_hardness_factor)
    tire_pressure = as_np_array(tire_pressure)
    tire_temperature = as_np_array(tire_temperature)
    tire_wear = as_np_array(tire_wear)
    camber = as_np_array(camber)

    # calculate the effects of the different factors
    base_friction = material_friction(tire_material_coeff, tread_amount, road_type, road_condition)
    hardness_factor = tire_hardness_effect(tire_hardness_factor)
    pressure_factor = tire_pressure_effect(tire_pressure)
    width_factor = tire_width_effect(tire_width)
    camber_factor = camber_effect(camber, tire_width)
    temperature_factor = temperature_effect(tire_temperature, tire_hardness_factor)
    tire_wear_factor = tire_wear_effect(tire_wear)

    # print("Raw values:")
    # print(
    #     f"{base_friction=}\n{hardness_factor=}\n{pressure_factor=}\n{width_factor=}\n{camber_factor=}\n{temperature_factor=}\n{tire_wear_factor=}"
    # )

    # Combine all factors to calculate the effective friction coefficient (aka "grip")
    grip_adjustment = weighted_sum(
        [
            hardness_factor,
            pressure_factor,
            width_factor,
            camber_factor,
            temperature_factor,
            tire_wear_factor,
        ],
        [
            TIRE_HARDNESS_EFFECT,
            TIRE_PRESSURE_EFFECT,
            TIRE_WIDTH_EFFECT,
            CAMBER_EFFECT,
            TEMPERATURE_EFFECT,
            TIRE_WEAR_EFFECT,
        ],
        print_labels=False,
        labels=[
            "hardness_factor",
            "pressure_factor",
            "width_factor",
            "camber_factor",
            "temperature_factor",
            "tire_wear_factor",
        ],
    )
    # print(f"Grip adjustment: {grip_adjustment}")
    return base_friction * grip_adjustment
