#!/usr/bin/env python3
import numpy
from matplotlib import pyplot as plt
from photograv import *


# Fly and get data
data = fly(
    px=2.6 * R_star_CenA,  # initial offset in px
    py=5000 * R_star_CenA,   # initial offset in py
    vx=0,  # initial speed in vx
    vy=-1000 * 13800,  # [m/sec], initial speed in vy; flying "downwards"
    ship_mass=.001 * 86,  # [kg]
    ship_sail_area=1162 * 86,  # [m^2], graphen for one gram
    M_star=M_star_CenA,
    R_star=R_star_CenA,
    L_star=L_star_CenA,
    minimum_distance_from_star=5 * R_star_CenA,  # only for display
    afterburner_distance=10e10,  # [R*] past star we can re-accelerate
    timestep=60 * 0.1,  # [minutes]
    number_of_steps=60000,
    return_mission=False)  # [to re-accelerate towards Earth]
print_flight_report(data)

# Output using Rene's file format. Headers are missing.
# numpy.savetxt("figure_1b.csv", data, delimiter=";")

# Make figure
scale = 100  # [stellar radii]
my_plot = make_figure_forces(data, scale, R_star_CenA)
my_plot.savefig("3a.pdf", bbox_inches='tight')
plt.show()
plt.close()

# Make figure
fig = plt.gcf()
scale = 15
my_plot = make_figure_distance_pitch_angle(data, scale, R_star_CenA)
my_plot.savefig("3b.pdf", bbox_inches='tight')
plt.show()
