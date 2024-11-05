# car-sim

Experimental engine for simulating the motion of a race car. Simulator-only, no visualization.

This module handles the core physics of the car, including acceleration, steering, centripetal force, dynamic vehicle load, and the effects of tire grip, temperature, pressure, wear, tread, camber, suspension stiffness and surface conditions.

This is not meant to be a simulator of real-world behavior! It is designed for sim games that simulate a lot of details. While it is definitely a lot more detailed than arcade games, it approximates a lot of behaviors. So it is not meant to be used as a physically-accurate simulator.

This module is meant to be usable in a variety of scenarios, so it intentionally does not contain any game or visual elements. You can use it for racing line simulations, machine learning, games (real-time or batch).

## Todo
Dynamics:
- tire temperature based on usage, tire hardness, road type, surface temperature
- tire health based on temperature and usage
- tire pressure based on temperature
- camber based on load and suspension stiffness