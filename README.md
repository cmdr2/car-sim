# car-sim

Experimental engine for simulating the motion of a race car. Simulator-only, no visualization.

This module handles the core physics of the car, including acceleration, steering, centripetal force, dynamic vehicle load, and the effects of tire grip, temperature, pressure, wear, tread, camber, suspension stiffness and surface conditions.

This is not meant to be a simulator of real-world behavior! It is designed for sim games that simulate a lot of details. While it is definitely a lot more detailed than arcade games, it approximates a lot of behaviors. So it is not meant to be used as a physically-accurate simulator.

This module is meant to be usable in a variety of scenarios, so it intentionally does not contain any game or visual elements. You can use it for racing line simulations, machine learning, games (real-time or batch).

## Implemented
Tire:
- grip ([code](https://github.com/cmdr2/car-sim/blob/main/carsim/tire/grip.py), [writeup](https://github.com/cmdr2/car-sim/wiki/Tire-Friction-Calculation))
- reaction force ([code](https://github.com/cmdr2/car-sim/blob/main/carsim/tire/force.py), [writeup](https://github.com/cmdr2/car-sim/wiki/Tire-Force-Calculation))

## Todo
Dynamics:
- tire temperature based on usage, tire hardness, road type, road temperature, slip (lateral and longitudinal)
- tire wear based on temperature, slip (lateral and longitudinal) and usage
- tire pressure based on temperature
- camber based on load and suspension stiffness