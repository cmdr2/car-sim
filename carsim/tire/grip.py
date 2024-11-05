import math
import numpy as np

from carsim.util import interpolate_curve, weighted_sum

OPTIMAL_CAMBER_FOR_REFERENCE_TIRE = -3.5  # Base optimal camber for reference width tires
REFERENCE_TIRE_WIDTH = 305  # Reference width in mm (standard tire width)
REFERENCE_OPTIMAL_PRESSURE = 28  # psi
REFERENCE_HARDNESS_TEMP_RANGES = (
    {  # https://web.archive.org/web/20240714102648/https://simracingsetup.com/f1-24/f1-24-ideal-tyre-temperatures/
        (0.9, 1.0): (85, 120),
        (0.8, 0.9): (85, 115),
        (0.7, 0.8): (80, 110),
        (0.6, 0.7): (80, 105),
        (0.5, 0.6): (75, 105),
        (0.4, 0.5): (55, 75),
        (0, 0.4): (45, 65),
    }
)
REFERENCE_TIRE_HEALTH_CURVE_POINTS = [
    (0, 0),
    (0.1, 0.03),
    (0.2, 0.08),
    (0.3, 0.2),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.75),
    (0.7, 0.82),
    (0.8, 0.9),
    (0.9, 0.95),
    (1.0, 1.0),
]
REFERENCE_TIRE_HEALTH_CURVE = interpolate_curve(REFERENCE_TIRE_HEALTH_CURVE_POINTS)

# Weightage of factors
SURFACE_FRICTION_EFFECT = 0.5
TIRE_HARDNESS_EFFECT = 1
TIRE_PRESSURE_EFFECT = 0.3
TIRE_WIDTH_EFFECT = 0.2
CAMBER_EFFECT = 0.2
TEMPERATURE_EFFECT = 0.5
TIRE_HEALTH_EFFECT = 1


# Friction based on material properties of tire and road
def material_friction(tire_material_coefficient, tread_amount, road_type, road_condition):
    """
    Calculate the base static friction coefficient based on tire and road material.

    :param tire_material_coeff: Static friction coefficient for tire material (μ_s).
    :param road_material_factor: Factor (between 0 and 1) representing the road surface conditions.
    :param surface_type: Type of surface. Accepted values: 'asphalt', 'concrete', 'dirt', 'gravel', 'ice'.
    :param tread_amount: Factor (0 to 1) representing the amount of tread. 0 is no tread (i.e. slicks).

    :return: Base static friction coefficient.
    """

    # Base friction coefficient for road types in perfect conditions
    road_base_friction = {"asphalt": 1.0, "concrete": 1.1, "dirt": 0.7, "gravel": 0.6, "grass": 0.5, "ice": 0.1}

    if road_type in road_base_friction:
        road_friction = road_base_friction[road_type] * pow(road_condition, 0.3)
    else:
        raise ValueError("Invalid road type")

    # Adjust tread effectiveness based on road type and condition
    if road_type in ["asphalt", "concrete"]:
        # On dry smooth surfaces, more tread reduces grip. But on dry wet/broken surfaces, more tread increases grip.
        x = road_condition - 0.5
        tread_effect = 1 - 1 * np.power(tread_amount, 0.5) * np.sign(x) * np.power(abs(x), 0.5)
        # print(f"{tread_effect=}")
    elif road_type in ["dirt", "gravel", "grass"]:
        # On loose surfaces, more tread increases grip
        tread_effect = 1 + 0.5 * tread_amount
    elif road_type == "ice":
        # On ice, tread has minimal effect
        tread_effect = 1

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
    hardness_effect = math.sqrt(1 - tire_hardness_factor)  # Soft tires increase friction

    return hardness_effect


def tire_pressure_effect(tire_pressure):
    """
    Calculate the effective friction based on tire pressure.

    :param tire_pressure: Tire pressure in psi (pounds per square inch).

    :return: Effective friction multiplier based on tire pressure.
    """
    # Tire pressure effect: Decrease friction as pressure increases
    pressure_effect = math.pow(REFERENCE_OPTIMAL_PRESSURE / tire_pressure, 0.7)

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
    return math.exp(-abs(camber - optimal_camber) / 5)  # Exponential decay for non-optimal camber


def get_temperature_range(tire_hardness_factor):
    """
    Returns the temperature range based on tire hardness.

    :param tire_hardness_factor (float): The tire hardness value.

    :return: tuple: The (low, high) temperature range for the given hardness.
    """
    for hardness_range, temp_range in REFERENCE_HARDNESS_TEMP_RANGES.items():
        if hardness_range[0] <= tire_hardness_factor <= hardness_range[1]:
            return temp_range
    return


def temperature_effect(tire_temperature, tire_hardness_factor):
    """
    Adjust friction based on temperature.

    :param tire_temperature (float): The current tire temperature in degrees Celsius.
    :param tire_hardness_factor (float): The tire hardness value.

    :return: float: Friction adjustment based on temperature.
    """
    low_optimal, high_optimal = get_temperature_range(tire_hardness_factor)

    # Cold tire (below optimal range)
    if tire_temperature < low_optimal:
        return math.pow(tire_temperature / low_optimal, 3)  # Decrease grip with temperature

    # Within optimal temperature range
    elif low_optimal <= tire_temperature <= high_optimal:
        return 1.0

    # Overheated tire (above optimal range)
    else:
        # Grip decreases linearly as the tire overheats
        overheating_factor = (tire_temperature - high_optimal) / (high_optimal - low_optimal)
        grip = max(0, 1 - overheating_factor)  # Decrease grip, but not below 0
        grip = pow(grip, 3)
        return grip


def tire_health_effect(tire_health):
    """
    Calculate the friction effect based on tire health.

    :param tire_health: Tire health as a value between 0 and 1 (0 = worn out, 1 = new).

    :return: Friction adjustment based on tire health.
    """
    tire_health = np.clip(tire_health, 0, 1)

    # Friction effect scales with tire health
    return REFERENCE_TIRE_HEALTH_CURVE(tire_health).item()


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
    tire_health,
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
    :param tire_health: Tire health as a value between 0 and 1 (0 = worn out, 1 = new).
    :param camber: Camber angle in degrees.

    :return: Effective grip
    """
    base_friction = material_friction(tire_material_coeff, tread_amount, road_type, road_condition)
    hardness_factor = tire_hardness_effect(tire_hardness_factor)
    pressure_factor = tire_pressure_effect(tire_pressure)
    width_factor = tire_width_effect(tire_width)
    camber_factor = camber_effect(camber, tire_width)
    temperature_factor = temperature_effect(tire_temperature, tire_hardness_factor)
    tire_health_factor = tire_health_effect(tire_health)

    # print(
    #     f"{base_friction=}, {hardness_factor=}, {pressure_factor=}, {width_factor=}, {camber_factor=}, {temperature_factor=}, {tire_health_factor=}"
    # )

    # Combine all factors to calculate the effective friction coefficient (aka "grip")
    grip_adjustment = weighted_sum(
        [
            hardness_factor,
            pressure_factor,
            width_factor,
            camber_factor,
            temperature_factor,
            tire_health_factor,
        ],
        [
            TIRE_HARDNESS_EFFECT,
            TIRE_PRESSURE_EFFECT,
            TIRE_WIDTH_EFFECT,
            CAMBER_EFFECT,
            TEMPERATURE_EFFECT,
            TIRE_HEALTH_EFFECT,
        ],
        print_labels=True,
        labels=[
            "hardness_factor",
            "pressure_factor",
            "width_factor",
            "camber_factor",
            "temperature_factor",
            "tire_health_factor",
        ],
    )
    # print(f"Grip adjustment: {grip_adjustment}")
    return base_friction * grip_adjustment


# # Example usage
# tire_material_coeff = 1.0  # Static friction coefficient for tire material (e.g., racing tire)
# tread_amount = 0  # Amount of tread on the tire. 0 is no tread (i.e. full slicks), and increasing tread towards 1
# road_type = "asphalt"
# road_condition = 0.8  # 0 is slippery, 1 is ideal
# tire_width = 305  #  mm
# tire_hardness_factor = 0.5
# tire_pressure = 30  # PSI
# tire_temperature = 90  # C
# tire_health = 0.5
# camber = -2.5  # degrees

# effective_grip = get_tire_grip(
#     tire_material_coeff,
#     tread_amount,
#     road_type,
#     road_condition,
#     tire_width,
#     tire_hardness_factor,
#     tire_pressure,
#     tire_temperature,
#     tire_health,
#     camber,
# )
# print(f"Effective Grip: {effective_grip:.3f}")
