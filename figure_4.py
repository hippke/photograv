import numpy
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
from photograv import *


fig = plt.figure()
plt.rc('text', usetex=False)
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2], sharey=ax1)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
fig.subplots_adjust(hspace=0.,wspace=0.)


######################################################################
# Cen A
# Fly and get data
data = fly(
    px = 6.0 * R_star_CenA,  # initial offset in px
    py = 5000 * R_star_CenA,   # initial offset in py
    vx = 0,  # initial speed in vx
    vy = -1000 * 13800,  # [m/sec], initial speed in vy; flying "downwards"
    ship_mass = .001,  # [kg]
    ship_sail_area = 1162,  # [m^2], graphen for one gram
    M_star = M_star_CenA,
    R_star = R_star_CenA,
    L_star = L_star_CenA,
    minimum_distance_from_star = 5 * R_star_CenA,  # only for display
    afterburner_distance = 10e10,  # [R*] past star we can re-accelerate
    timestep = 60 * 0.5,  # [minutes]
    number_of_steps = 15000,
    return_mission = False)  # [to re-accelerate towards Earth]

print_flight_report(data)

# Find closest encounter
encounter_time, step_of_closest_encounter = get_closest_encounter(data)

time = data['time'] / 3600 - encounter_time
speed = data['ship_speed']
ax1.plot(time, speed, color='black', linewidth=0.5)
ax1.plot([0, 0], [0, 14000],
    linestyle = 'dotted', color='black', linewidth=0.5)


######################################################################
# Cen B
# Fly and get data
data = fly(
    px = 4.2 * R_star_CenB,  # initial offset in px
    py = 5000 * R_star_CenB,   # initial offset in py
    vx = 0,  # initial speed in vx
    vy = -1000 * 7290,  # [m/sec], initial speed in vy; flying "downwards"
    ship_mass = .001,  # [kg]
    ship_sail_area = 1162,  # [m^2], graphen for one gram
    M_star = M_star_CenB,
    R_star = R_star_CenB,
    L_star = L_star_CenB,
    minimum_distance_from_star = 5 * R_star_CenB,  # only for display
    afterburner_distance = 10e20,  # [R*] past star we can re-accelerate
    timestep = 60 * 1,  # [minutes]
    number_of_steps = 20000,
    return_mission = False)  # [to re-accelerate towards Earth]

# Find closest encounter
encounter_time, step_of_closest_encounter = get_closest_encounter(data)

time = data['time'] / 3600 - encounter_time
speed = data['ship_speed']
ax2.plot(time, speed, color='black', linewidth=0.5)
ax2.plot([0, 0], [0, 14000],
    linestyle = 'dotted', color='black', linewidth=0.5)


######################################################################
# Cen C
# Fly and get data
data = fly(
    px = 3.62 * R_star_CenC,  # initial offset in px
    py = 5000 * R_star_CenC,   # initial offset in py
    vx = 0,  # initial speed in vx
    vy = -1000 * 1110,  # [m/sec], initial speed in vy; flying "downwards"
    ship_mass = .001,  # [kg]
    ship_sail_area = 1162,  # [m^2], graphen for one gram
    M_star = M_star_CenC,
    R_star = R_star_CenC,
    L_star = L_star_CenC,
    minimum_distance_from_star = 1 * R_star_CenC,  # only for display
    afterburner_distance = 10e10,  # [R*] past star we can re-accelerate
    timestep = 60 * 5,  # [minutes]
    number_of_steps = 5000,
    return_mission = False)  # [to re-accelerate towards Earth]
print_flight_report(data)


# Find closest encounter
encounter_time, step_of_closest_encounter = get_closest_encounter(data)

time = data['time'] / 3600 - encounter_time
speed = data['ship_speed']
ax3.plot(time, speed, color='black', linewidth=0.5)
ax3.plot([0, 0], [0, 14000],
    linestyle = 'dotted', color='black', linewidth=0.5)


# PLOT
######################################################################
scale = 5
minor_xticks = numpy.arange(-scale, scale, scale/10)

minor_yticks = numpy.arange(0, 14000, 400)
ax1.set_xticks(minor_xticks, minor=True)
ax1.set_yticks(minor_yticks, minor=True)
ax1.get_yaxis().set_tick_params(direction='in')
ax1.get_xaxis().set_tick_params(direction='in')
ax1.get_yaxis().set_tick_params(which='both', direction='in')
ax1.get_xaxis().set_tick_params(which='both', direction='in')
ax1.set_xlim(-scale, +scale)
ax1.set_ylim(1000, 5000)

ax2.set_xticks(minor_xticks, minor=True)
ax2.set_yticks(minor_yticks, minor=True)
ax2.get_yaxis().set_tick_params(direction='in')
ax2.get_xaxis().set_tick_params(direction='in')
ax2.get_yaxis().set_tick_params(which='both', direction='in')
ax2.get_xaxis().set_tick_params(which='both', direction='in')
ax2.set_xlim(-scale, +scale)
ax2.set_ylim(1000, 5000)

ax3.set_xticks(minor_xticks, minor=True)
ax3.set_yticks(minor_yticks, minor=True)
ax3.get_yaxis().set_tick_params(direction='in')
ax3.get_xaxis().set_tick_params(direction='in')
ax3.get_yaxis().set_tick_params(which='both', direction='in')
ax3.get_xaxis().set_tick_params(which='both', direction='in')
ax3.set_xlim(-scale, +scale)
ax3.set_ylim(1000, 5000)

# Minor y-tick marks every 500 km/sec
minor_locator = AutoMinorLocator(4)
ax1.yaxis.set_minor_locator(minor_locator)
ax2.yaxis.set_minor_locator(minor_locator)
ax3.yaxis.set_minor_locator(minor_locator)

plt.xlabel('Time Around Closest Approach [Hours]', x=-0.5)
fig.text(0.04, 0.5, 'Speed [km/s]', fontweight='bold', va='center', rotation='vertical')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlim(-scale, +scale)
plt.ylim(0, 14000)

fig.text(0.27, 0.87, r'$\alpha$ Cen A', fontweight='bold', va='center')
fig.text(0.53, 0.87, r'$\alpha$ Cen B', fontweight='bold', va='center')
fig.text(0.80, 0.87, r'$\alpha$ Cen C', fontweight='bold', va='center')

plt.savefig("figure_4.pdf", bbox_inches='tight')
plt.show()
