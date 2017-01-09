#!/usr/bin/env python3
from matplotlib import pyplot as plt
from photograv import *


# Fly and get data
data = fly(
    px=2.80 * R_star_CenA,  # initial offset in px
    py=10 * AU,   # initial offset in py
    vx=0,  # initial speed in vx
    vy=-1000 * 1270,  # [m/sec], initial speed in vy; flying "downwards"
    ship_mass = 0.001,  # [kg]
    ship_sail_area = 10,  #[m^2]
    M_star = M_star_CenA,
    R_star = R_star_CenA,
    L_star = L_star_CenA,
    minimum_distance_from_star = 5 * R_star_CenA,  # only for display
    afterburner_distance = 10e10,  # past star we can re-accelerate
    timestep = 60 * 5,  # [minutes]
    number_of_steps = 10000,
    return_mission = False)  # [to re-accelerate towards Earth]
print_flight_report(data)


# We can save the output in René's format (without the headers):
#numpy.savetxt("figure_1b.csv", data, delimiter=";")

# Make figure
fig = plt.gcf()
my_plot = make_figure_flight(
    data = data,
    stellar_radius = R_star_CenA,
    scale = 20,  # of plot in [stellar radii]
    flight_color='black',  # color of flight trajectorie line
    redness = 0.7,  # 1:red star, 0:yellow; quad limb is darkening hard-coded
    show_burn_circle = False,  # show dashed circle for minimum distance
    star_name = r'$\alpha$ Cen A',
    weight_ratio = 0.1,  # to print 0.1g (we have 10m^2 and 1g)
    circle_spacing_minutes = 120,  # grid of circle marks every N minutes
    annotate_cases = False,  # I, II, III, IV, V
    caption = 'b')  # Figure "suptitle"
fig.savefig("2b.pdf", bbox_inches = 'tight')
plt.show()
