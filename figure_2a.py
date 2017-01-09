#!/usr/bin/env python3
import numpy
from matplotlib import pyplot as plt
from photograv import *

# Define spaceship:
number_of_steps = 300000  # 300000
timestep = 60 * 0.1  # 0.1 One timestep every 5 minutes
ship_sail_area = 10  # sail surface im square meters. Weight is 1g hard-coded.
afterburner_distance = 10e10  # [R_star]
speed = 1260 # [km/sec], initial speed of spaceship
minimum_distance_from_star = 5 * R_star_CenA  # closest distance to star. Only used to check, not to adjust sim!

ship_mass = .001  # [kg]
ship_py = 10 * AU  # start position vertical / distance travelled
ship_vx = 0
ship_vy = -speed * 1000  # unit conversion; sign: fly downwards


# Define horizontal offsets for the series of curves
offsets = numpy.arange(0., 20, 0.1)
offsets = offsets * R_star_CenA
scale = 20

fig = plt.gcf()
return_mission = True
my_plot = make_figure_series_of_curves(ship_py, ship_vx, ship_vy, ship_mass, ship_sail_area,
           M_star_CenA, R_star_CenA, L_star_CenA, minimum_distance_from_star, afterburner_distance,
           timestep, number_of_steps, return_mission,
           offsets, scale, caption = 'a', colorbar = False)

fig.savefig("2a.pdf", bbox_inches = 'tight')
plt.show()
