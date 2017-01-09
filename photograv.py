#!/usr/bin/env python3
import numpy
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
from numpy import arctan, arctan2, sqrt, pi, sin, cos, arccos, radians
from scipy import optimize


"""Provide constants"""

G = 6.67428e-11  # the gravitational constant G
c = 299792458  # [m/sec] speed of light
AU = (149.6e6 * 1000)  # [m] 149.6 million km

# Sun
sun_radius = 695700000  # [m]
sun_mass = 1.989 * 10**30  # [kg]
sun_luminosity = 3.86 * 10**26  # [Watt] stellar luminosity

# CenA:
L_star_CenA = sun_luminosity * 1.522
R_star_CenA = sun_radius * 1.224
M_star_CenA = sun_mass * 1.105

# CenB:
L_star_CenB = sun_luminosity * 0.503
R_star_CenB = sun_radius * 0.863
M_star_CenB = sun_mass * 0.934

# CenC:
L_star_CenC = sun_luminosity * 138 * 10e-6
R_star_CenC = sun_radius * 0.145
M_star_CenC = sun_mass * 0.123


def quadratic_limb_darkening(impact, limb1, limb2):
    "Quadratic limb darkening. Kopal 1950, Harvard Col. Obs. Circ., 454, 1"
    impact = cos(1 - impact)
    return 1 - limb1 * (1 - impact) - limb2 * (1 - impact) ** 2


def get_gravity_force(px_ship, py_ship, mass_ship, mass_star):
    """Return the gravity force in x and y directions"""

    distance = sqrt(px_ship**2 + py_ship**2)
    force = G * mass_ship * mass_star / (distance**2)

    # direction of the force
    theta = arctan2(py_ship, px_ship)

    fx = -cos(theta) * force
    fy = -sin(theta) * force
    return fx, fy


def get_photon_force(x, y, vx, vy, R_star, L_star, ship_sail_area):
    """Return the photon force in x and y directions"""

    r = sqrt((x**2) + (y**2))  # distance between sail and x,y=(0,0) in m

    if x != 0:  # prevent division by zero if float x == 0
        phi = arctan(y / x)  # angle in radians
    else:
        phi = 0

    def get_F_alpha_r(alpha):
        return L_star * ship_sail_area / (3 * pi * c * R_star**2) * \
            (1 - (1 - (R_star / r)**2)**(3 / 2.)) * cos(alpha)

    def photon_function(alpha):
        F_alpha_r = get_F_alpha_r(alpha)
        F_x = F_alpha_r * cos(alpha + phi)
        F_y = F_alpha_r * sin(alpha + phi)
        nu = arccos((F_x * vx + F_y * vy) / \
            (sqrt(F_x**2 + F_y**2) * sqrt(vx**2 + vy**2)))
        F_x_final = F_alpha_r * cos(nu)
        return F_x_final

    alpha_min = optimize.fmin_slsqp(
        photon_function,
        bounds=[(-0.5 * pi, +0.5 * pi)],
        x0=0,
        disp=0)

    # Calculate F_x and F_y for best alpha_min
    F_alpha_r = get_F_alpha_r(alpha_min)
    F_x = F_alpha_r * numpy.cos(alpha_min + phi)
    F_y = F_alpha_r * numpy.sin(alpha_min + phi)
    return F_x[0], F_y[0], alpha_min


def fly(
    px,
    py,
    vx,
    vy,
    ship_mass,
    ship_sail_area,
    M_star,
    R_star,
    L_star,
    minimum_distance_from_star,
    afterburner_distance,
    timestep,
    number_of_steps,
    return_mission =False):
    """Loops through the simulation, returns result array"""

    # Data return array
    result_array = numpy.zeros((number_of_steps,), dtype=[
                               ('step', 'int32'),
                               ('time', 'int32'),
                               ('px', 'f8'),
                               ('py', 'f8'),
                               ('F_gravity', 'f8'),
                               ('F_photon', 'f8'),
                               ('photon_acceleration', 'f8'),
                               ('ship_speed', 'f8'),
                               ('alpha', 'f8'),
                               ('stellar_distance', 'f8'),
                               ('sail_not_parallel', 'bool')])
    result_array[:] = numpy.NAN
    deceleration_phase = True

    # Main loop
    for step in range(number_of_steps):

        # Gravity force
        gravity_F_x, gravity_F_y = get_gravity_force(
            px, py, ship_mass, M_star)
        vx += gravity_F_x / ship_mass * timestep
        vy += gravity_F_y / ship_mass * timestep

        # Check distance from star
        star_distance = sqrt((px / R_star) ** 2 + (py / R_star) ** 2)

        # Check if we are past closest encounter. If yes, switch sail off
        previous_distance = result_array['stellar_distance'][step-1]
        if step > 2 and star_distance > previous_distance:
            deceleration_phase = False

        # Check if we are past the star and at afterburner distance
        # If yes, switch sail on again
        if not return_mission and py < 0 and star_distance > afterburner_distance:
            deceleration_phase = True  # Actually acceleration now!

        # Inferno time: If we are inside the star, the simulation ends
        if star_distance < 1.:
            print('Exit due to ship being inside the star.')
            break

        # In case we are inside the minimum distance, the simulation ends
        if star_distance < minimum_distance_from_star / R_star:
            print('Exit due to ship being inside the minimum distance')
            break

        # Special case return mission
        # This is an ugly hardcoded special case. To be optimized in the future
        if return_mission and px < 0 and py / R_star > 4.75:
            deceleration_phase = True  # Actually acceleration now!

        # Photon pressure force
        photon_F_x, photon_F_y, current_alpha = get_photon_force(
            px, py, vx, vy, R_star, L_star, ship_sail_area)

        # Set sign of photon force
        if deceleration_phase:
            if px > 0:
                vx += +photon_F_x / ship_mass * timestep
                vy += +photon_F_y / ship_mass * timestep
            else:
                vx += -photon_F_x / ship_mass * timestep
                vy += -photon_F_y / ship_mass * timestep

        # Update positions
        px += vx * timestep
        py += vy * timestep

        """Calculate interesting values"""

        # If we do not decelerate: sail shall be parallel with zero photon force
        if not deceleration_phase:
            photon_F_x = 0
            photon_F_y = 0

        # Acceleration due to photon pressure only
        a_photon_x = photon_F_x / ship_mass
        a_photon_y = photon_F_y / ship_mass
        a_photon_tot = sqrt(a_photon_x**2 + a_photon_y**2)

        # Forces
        photon_force = sqrt(photon_F_x ** 2 + photon_F_y ** 2)
        gravity_force = sqrt(gravity_F_x ** 2 + gravity_F_y ** 2)
        probe_velocity = sqrt(abs(vx) ** 2 + abs(vy) ** 2)

        # Write interesting values into return array
        result_array['step'][step] = step
        result_array['time'][step] = step * timestep
        result_array['px'][step] = px / sun_radius
        result_array['py'][step] = py / sun_radius
        result_array['F_gravity'][step] = gravity_force
        result_array['F_photon'][step] = photon_force
        result_array['photon_acceleration'][step] = a_photon_tot
        result_array['ship_speed'][step] = probe_velocity/1000.
        result_array['alpha'][step] = current_alpha
        result_array['stellar_distance'][step] = star_distance
        result_array['sail_not_parallel'][step] = deceleration_phase

    return result_array


def print_flight_report(data):
    print('speed [km/sec] start', '{:10.1f}'.format(data['ship_speed'][1]),
        ' - speed end', '{:10.1f}'.format(data['ship_speed'][-1]))
    print('Overall closest encounter to star [stellar radii]',
        '{:1.3f}'.format(numpy.amin(data['stellar_distance'])))

    index_min = numpy.argmin(data['stellar_distance'])
    print('Closest encounter at time [min]',
        '{:1.3f}'.format(numpy.amin(data['time'][index_min] / 60)))
    print('Closest encounter at speed [km/s]',
        '{:1.3f}'.format(numpy.amin(data['ship_speed'][index_min])))

    angle = abs(arctan2(data['py'][-1], data['px'][-1]) * (360 / (2 * pi))) - 90
    print('Deflection angle [degree]',
        '{:1.1f}'.format(angle), '{:1.1f}'.format(180 - angle))
    print('Total time range is [seconds]',
        data['time'][0], 'to', data['time'][-1])
    has_sail_switched_off = False
    print(len(data['time']))
    for step in range(len(data['time'])):
        speed_change = data['ship_speed'][step] - data['ship_speed'][step-1]  # km/sec
        time_change = data['time'][step] - data['time'][step-1]  # sec
        g_force = speed_change / time_change * 1000 / 9.81

        total_force = data['F_photon'][step] - data['F_gravity'][step]
        gee = total_force / 0.086 / 9.81
        # print(step, data['stellar_distance'][step], data['ship_speed'][step], speed_change, time_change, g_force, gee)


        if data['sail_not_parallel'][step] == False:
            has_sail_switched_off = True
            percentage_when_off = data['time'][step] / data['time'][-1] * 100
            #print('Sail switched off at time [seconds]',
                #data['time'][step], '{:10.1f}'.format(percentage_when_off), '%')
            #break
    if not has_sail_switched_off:
        print('Sail was always on')


def make_figure_flight(
        data,
        stellar_radius,
        scale,
        flight_color,
        redness,
        show_burn_circle,
        star_name,
        weight_ratio,
        circle_spacing_minutes,
        annotate_cases,
        caption,
        colorbar=True):
    fig = plt.gcf()
    ax = fig.add_subplot(111, aspect='equal')

    # Flight trajectorie
    px_stellar_units = data['px'] * sun_radius / stellar_radius
    py_stellar_units = data['py'] * sun_radius / stellar_radius
    plt.plot(
        px_stellar_units,
        py_stellar_units,
        color=flight_color,
        linewidth=0.5)

    # G-Forces

    cNorm = matplotlib.colors.Normalize(vmin=-11.5, vmax=2)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='Reds_r')

    last_label = 18.5

    if colorbar:

        for step in range(len(data['time'])):
            # Color bar
            x1 = 15
            x2 = 20
            y1 = data['py'][step] * sun_radius / stellar_radius
            y2 = y1
            if -20 < y1 < 20:
                speed_change = data['ship_speed'][step] - data['ship_speed'][step-1]  # km/sec
                time_change = data['time'][step] - data['time'][step-1]  # sec
                g_force = speed_change / time_change * 1000 / 9.81
                colorVal = scalarMap.to_rgba(g_force)
                plt.plot([x1, x2], [y1, y2], color = colorVal, linewidth = 4.)
            if y1 < last_label + 1 and y1 < 18 and y1 > -18:
                if g_force < -10:
                    textcolor = 'white'
                else:
                    textcolor = 'black'
                text = r'${:2.1f} g$'.format(g_force)
                plt.annotate(
                    text, xy=(17, y1),
                    fontsize=12,
                    horizontalalignment='center',
                    color=textcolor)

                last_label = last_label - 4

    # If desired, show dashed circle for minimum distance
    if show_burn_circle:
        circle2 = plt.Circle(
            (0, 0),
            5,
            color='black',
            fill=False,
            linestyle='dashed',
            linewidth=0.5)  # dashed version
        # shaded version
        # circle2=plt.Circle((0,0), 5, color=(0.9, 0.9, 0.9), fill=True)

        fig.gca().add_artist(circle2)

    # Star in the center with limb darkening
    star_quality = 50  # number of shades
    limb1 = 0.4703  # quadratic limb darkening parameters
    limb2 = 0.236
    for i in range(star_quality):
        impact = i / float(star_quality)
        LimbDarkening = quadratic_limb_darkening(impact, limb1, limb2)
        Sun = plt.Circle(
            (0, 0),
            1 - i / float(star_quality),
            color=(LimbDarkening, redness * LimbDarkening, 0))
        plt.gcf().gca().add_artist(Sun)

    # Calculate and print deflection angle (and star name) if desired
    if star_name != '':
        ax.annotate(star_name, xy=(-2.5, 1.75))
        deflection_angle = abs(
            arctan2(data['py'][-1], data['px'][-1]) * (360 / (2 * pi))) - 90

        # print angle
        text_deflection_angle = r'$\delta = {:1.0f} ^\circ$'.format(deflection_angle)
        ax.annotate(
            text_deflection_angle,
            xy=(-scale + 1, scale - 2.5),
            fontsize=16)

        # Add a circle mark at the closest encounter
        index_min = numpy.argmin(data['stellar_distance'])
        min_x_location = data['px'][index_min] * sun_radius / stellar_radius
        min_y_location = data['py'][index_min] * sun_radius / stellar_radius
        marker = plt.Circle(
            (min_x_location, min_y_location), 0.2, color='red', fill=False)
        plt.gcf().gca().add_artist(marker)

        # print weight ratio
        text_weight_ratio = r'$\sigma = {:1.1f}$ g/m$^2$'.format(weight_ratio)
        ax.annotate(text_weight_ratio, xy=(-scale + 1, scale - 5), fontsize=16)

        # print entry speed
        text_entry_speed = r'$\nu_{{\infty}} = {:10.0f}$ km/s'.format(data['ship_speed'][1])
        ax.annotate(text_entry_speed, xy=(-scale + 1, scale - 8), fontsize=16)

        # Add circle marks every [circle_spacing_minutes] and annotate them
        current_marker_number = 0
        it = numpy.nditer(data['time'], flags=['f_index'])
        while not it.finished:
            if (it[0] / 60) % circle_spacing_minutes == 0:
                x_location = data['px'][it.index] * sun_radius / stellar_radius
                y_location = data['py'][it.index] * sun_radius / stellar_radius
                speed = data['ship_speed'][it.index]

                # Check if inside the plotted figure (faster plot generation)
                if abs(x_location) < scale and abs(y_location) < scale:

                    # Yes, mark it
                    marker = plt.Circle(
                        (x_location, y_location),
                        0.1,
                        color='black',
                        fill=True)
                    plt.gcf().gca().add_artist(marker)
                    current_marker_number = current_marker_number + 1

                    # Add sail with angle as marker

                    angle = data['alpha'][it.index]
                    if y_location > 0:
                        angle = -angle

                    # Sail is parallel, put it in direction of star:
                    if not data['sail_not_parallel'][it.index]:
                        v1_theta = arctan2(0, 0)
                        v2_theta = arctan2(y_location, x_location)
                        angle = (v2_theta - v1_theta) * (180.0 / pi)
                        angle = radians(angle)

                    # Calculate start and end positions of sail line
                    length = 1
                    endy = y_location + (length * sin(angle))
                    endx = x_location + (length * cos(angle))
                    starty = y_location - (length * sin(angle))
                    startx = x_location - (length * cos(angle))
                    plt.plot([startx, endx], [starty, endy], color='black')

                    # Make arrow between first and second circle mark
                    if current_marker_number == 1:
                        ax.arrow(
                            x_location,
                            y_location - 2, 0.0,
                            -0.01,
                            head_width=0.75,
                            head_length=0.75,
                            fc='k',
                            ec='k',
                            lw=0.01)

                    # To avoid crowding of text labels, set them sparsely
                    if data['ship_speed'][it.index] > 300:
                        n_th_print = 1  # print all labels
                    if 150 < data['ship_speed'][it.index] < 300:
                        n_th_print = 3  # print every 5th label
                    if data['ship_speed'][it.index] < 150:
                        n_th_print = 5  # print every 5th label

                    if -19 < y_location < 19 and (it[0] / 60) % (circle_spacing_minutes * n_th_print) == 0:
                        test_speed = ''
                        text_speed = r'{:10.0f} km/s'.format(speed)
                        ax.annotate(
                            text_speed,
                            xy=(x_location + 1, y_location - 0.5),
                            fontsize=12)

            it.iternext()

    # Format the figure
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    minor_ticks = numpy.arange(-scale, scale, 1)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)
    plt.xlim(-scale, scale)
    plt.ylim(-scale, scale)
    plt.xlabel('Distance [Stellar Radii]', fontweight='bold')
    plt.ylabel('Distance [Stellar Radii]', fontweight='bold')

    if caption != '':
        fig.suptitle(
            caption,
            fontsize=16,
            fontweight='bold',
            weight='heavy',
            x=0.22,
            y=0.95)

    if annotate_cases:
        # Annotate cases
        ax.annotate('I, II', xy=(-1, 3))
        ax.annotate('III', xy=(-16, -5))
        ax.annotate('IV', xy=(6, -18))

    return plt


def make_figure_speed(data, scale, caption):

    encounter_time, step_of_closest_encounter = get_closest_encounter(data)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    minor_xticks = numpy.arange(-scale, scale, scale/10)
    ax.set_xticks(minor_xticks, minor=True)
    minor_yticks = numpy.arange(0, 14000, 400)
    ax.set_yticks(minor_yticks, minor=True)
    ax.get_yaxis().set_tick_params(direction='in')
    ax.get_xaxis().set_tick_params(direction='in')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.suptitle(caption, fontsize=14, fontweight='bold', x=0.131, y=0.95)
    if caption == 'a':
        ax.annotate(r'$\alpha$ Cen A', xy=(2, 12000))
    if caption == 'b':
        ax.annotate(r'$\alpha$ Cen B', xy=(2, 12000))
    if caption == 'c':
        ax.annotate(r'$\alpha$ Cen C', xy=(2, 12000))

    time = data['time'] / 3600 - encounter_time
    speed = data['ship_speed']
    plt.plot(time, speed, color='black', linewidth=0.5)

    # Vertical line
    plt.plot(
        [0, 0],
        [0, 14000],
        linestyle='dotted',
        color='black',
        linewidth=0.5)

    plt.xlim(-scale, +scale)
    plt.ylim(0, 14000)
    plt.xlabel('Time Around Closest Approach [Hours]', fontweight='bold')
    plt.ylabel('Speed [km/s]', fontweight='bold')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    return plt


def get_closest_encounter(data):
    closest_encounter = float("inf")
    step_of_closest_encounter = 0
    for step in range(len(data['stellar_distance'])):
        if data['stellar_distance'][step] < closest_encounter:
            closest_encounter = data['stellar_distance'][step]
            step_of_closest_encounter = step
    encounter_time = data['time'][step_of_closest_encounter] / 3600  # hours
    print('Closest encounter at step', step_of_closest_encounter,
          'with distance [stellar radii]', '{:1.1f}'.format(closest_encounter),
          'at time', '{:1.1f}'.format(encounter_time))

    return encounter_time, step_of_closest_encounter


def make_figure_forces(data, scale, stellar_radius):

    # Find closest encounter
    encounter_time, step_of_closest_encounter = get_closest_encounter(data)

    # select data from start to closest encounter
    px_stellar_units = data['px'] * sun_radius / stellar_radius
    py_stellar_units = data['py'] * sun_radius / stellar_radius

    # distance between sail and origin in [m]
    distance = sqrt((px_stellar_units**2) + (py_stellar_units**2))
    distance[:step_of_closest_encounter] = -distance[:step_of_closest_encounter]

    total_force = data['F_photon'] - data['F_gravity']
    gee = total_force / 0.086 / 9.81
    print('max total_force', numpy.max(total_force))
    print('max gee', numpy.max(gee))

    # Figure
    fig = plt.figure(figsize=(6, 6))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.gca()

    minor_xticks = numpy.arange(-scale, scale, 5)
    ax.set_xticks(minor_xticks, minor=True)
    ax.get_xaxis().set_tick_params(which='both', direction='in')

    fig.suptitle('a', fontsize=14, fontweight='bold', x=0.131, y=0.95)
    time = data['time'] / 3600 - encounter_time
    total_force = data['F_photon'][:step_of_closest_encounter]
    - data['F_gravity'][:step_of_closest_encounter]
    ax.plot(distance[:step_of_closest_encounter],
        data['F_gravity'][:step_of_closest_encounter],
        color='black', linewidth=1.)
    ax.plot(distance[step_of_closest_encounter:],
        data['F_gravity'][step_of_closest_encounter:],
        color='black', linewidth=1.)
    ax.plot(distance[:step_of_closest_encounter],
        data['F_photon'][:step_of_closest_encounter],
        color='black', linestyle = 'dashed', linewidth=1.)
    offset = 0.02  # to show photon and total lines separatley
    ax.plot(distance[:step_of_closest_encounter],
        total_force-(offset*total_force), color='red', linewidth=1.)

    # Vertical line
    ax.plot(
        [0, 0],
        [0, numpy.amax(total_force * 1.1)],
        linestyle='dotted',
        color='black',
        linewidth=0.5)

    ax.annotate('Photon', xy=(-90, 12), rotation=20, color='black')
    ax.annotate('Total', xy=(-90, 4), rotation=20, color='red')
    ax.annotate('Gravity', xy=(-90, 2.1e-3), rotation=20, color='black')
    ax.yaxis.tick_left()
    ax.set_xlabel('Stellar Distance [Stellar Radii]', fontweight='bold')
    ax.set_ylabel('Force Acting on the Sail [Newton]', fontweight='bold')
    ax.set_xlim(-scale, + scale)
    ax.set_ylim(10e-4, numpy.amax(total_force * 1.2))
    ax.set_yscale('log')

    # Right axis: g-force
    ax2 = fig.add_subplot(111, sharex=ax, frameon=False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlim(-scale, + scale)
    ax2.set_ylim(5, numpy.max(gee) * 1.2)
    ax2.set_yscale('log')
    ax2.set_ylabel(
        r'Deceleration of the Sail [$g=9.81$m/s$^2$]', fontweight='bold')
    plt.yscale('log')

    # Inset with zoom
    ax1 = fig.add_subplot(111)
    axins = inset_axes(
        ax,
        1.75,
        1.75,
        loc=2,
        bbox_to_anchor=(0.56, 0.89),
        bbox_transform=ax.figure.transFigure)  # no zoom
    offset = 0.001  # to show photon and total lines separatley

    axins.plot(
        distance[:step_of_closest_encounter],
        data['F_photon'][:step_of_closest_encounter],
        color='black',
        linewidth=1,
        linestyle='dashed')
    axins.plot(
        distance[:step_of_closest_encounter],
        total_force - (offset * total_force),
        color='red',
        linewidth=1)

    # Vertical line
    axins.plot(
        [0, 0],
        [0, numpy.amax(total_force * 1.1)],
        linestyle='dotted',
        color='black',
        linewidth=0.5)
    axins.set_xlim(-5.5, -5)
    axins.set_ylim(1200, 1400)

    return plt


def make_figure_distance_pitch_angle(data, scale, stellar_radius):

    encounter_time, step_of_closest_encounter = get_closest_encounter(data)

    # select data from start to closest encounter
    stellar_radii = sun_radius / stellar_radius
    px_stellar_units = data['px'][:step_of_closest_encounter] * stellar_radii
    py_stellar_units = data['py'][:step_of_closest_encounter] * stellar_radii
    pitch_angle = data['alpha'][:step_of_closest_encounter]
    pitch_angle = -numpy.degrees(pitch_angle)

    # distance between sail and origin in [m]
    distance = sqrt((px_stellar_units**2) + (py_stellar_units**2))

    # make figure
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    fig.suptitle('b', fontsize=14, fontweight='bold', x=0.131, y=0.95)
    plt.plot(distance, pitch_angle, linewidth=0.5, color='black')
    plt.xscale('log')
    ax.set_xticks([5, 10, 100])
    ax.set_xticklabels(["50", "10", "100"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim(4, 100)
    plt.ylim(-45, 0)
    plt.xlabel('Stellar Distance [Stellar Radii]', fontweight='bold')
    plt.ylabel(
        r'Sail Pitch Angle $\alpha$ During Approach [Degrees]',
        fontweight='bold')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    return plt


def make_figure_series_of_curves(
    ship_py,
    ship_vx,
    ship_vy,
    ship_mass,
    ship_sail_area,
    M_star,
    R_star,
    L_star,
    minimum_distance_from_star,
    afterburner_distance,
    timestep,
    number_of_steps,
    return_mission,
    offsets,
    scale,
    caption,
    colorbar=False):

    color = 'black'
    redness = 0.7
    star_name = ''
    weight_ratio = 1 / ship_sail_area
    circle_spacing_minutes = 2e10  # never
    show_burn_circle = True
    annotate_cases = True

    # Special case, print one extra to show return mission
    """
    if return_mission:

        print('Generating special case return mission')
        horizontal_offset = 2.7 * R_star
        my_data = fly(horizontal_offset, ship_py, ship_vx, ship_vy, ship_mass,
            ship_sail_area, M_star, R_star, L_star, minimum_distance_from_star, afterburner_distance, timestep, number_of_steps, return_mission=True)
        print_flight_report(my_data)

        my_figure = make_figure_flight(my_data, R_star, scale, color,
            redness, show_burn_circle, star_name, weight_ratio, circle_spacing_minutes, annotate_cases, caption)

        print('Generating special case: bound orbit')
        horizontal_offset = 2.7 * R_star
        afterburner_distance_special = 10e10
        my_data = fly(horizontal_offset, ship_py, ship_vx, ship_vy, ship_mass,
            ship_sail_area, M_star, R_star, L_star, minimum_distance_from_star, afterburner_distance_special, timestep, number_of_steps, return_mission=False)
        print_flight_report(my_data)

        my_figure = make_figure_flight(my_data, R_star, scale, color,
            redness, show_burn_circle, star_name, weight_ratio, circle_spacing_minutes, annotate_cases, caption)

        print('Generating special case: break out to -x, -y')
        horizontal_offset = 2.7 * R_star
        afterburner_distance_special = 14
        my_data = fly(horizontal_offset, ship_py, ship_vx, ship_vy, ship_mass,
            ship_sail_area, M_star, R_star, L_star, minimum_distance_from_star, afterburner_distance_special, timestep, number_of_steps, return_mission=False)
        print_flight_report(my_data)

        my_figure = make_figure_flight(my_data, R_star, scale, color,
            redness, show_burn_circle, star_name, weight_ratio, circle_spacing_minutes, annotate_cases, caption)
    """

    annotate_cases = False  # Avoid printing it many times
    start = numpy.min(offsets)
    stop = numpy.max(offsets)
    iterations = numpy.size(offsets)
    iter_counter = 0
    for horizontal_offset in offsets:
        print(
            'Now running iteration', iter_counter + 1,
            'of', int(iterations),
            'with offset', horizontal_offset / R_star)
        iter_counter += 1
        my_data = fly(
            horizontal_offset,
            ship_py,
            ship_vx,
            ship_vy,
            ship_mass,
            ship_sail_area,
            M_star,
            R_star,
            L_star,
            minimum_distance_from_star,
            afterburner_distance,
            timestep,
            number_of_steps,
            return_mission=False)

        print_flight_report(my_data)
        color_shade = iter_counter / iterations
        if color_shade > 1:
            color_shade = 1
        if color_shade < 0:
            color_shade = 0
        color = [1 - color_shade, 0, color_shade]

        my_figure = make_figure_flight(
            my_data,
            R_star,
            scale,
            color,
            redness,
            show_burn_circle,
            star_name,
            weight_ratio,
            circle_spacing_minutes,
            annotate_cases,
            caption='',
            colorbar=False)

    fig = plt.gcf()
    ax = fig.add_subplot(111, aspect='equal')
    ax.annotate('I, II', xy=(-1, 3))
    ax.annotate('III', xy=(-16, -5))
    ax.annotate('IV', xy=(6, -18))
    fig.suptitle('a', fontsize=16, fontweight='bold', weight='heavy', x=0.22, y=0.95)

    return my_figure


def make_video_flight(
        data,
        stellar_radius,
        scale,
        flight_color,
        redness,
        show_burn_circle,
        star_name,
        weight_ratio,
        circle_spacing_minutes,
        annotate_cases,
        caption):
    fig = plt.gcf()
    ax = fig.add_subplot(111, aspect='equal')

    # Flight trajectorie
    px_stellar_units = data['px'] * sun_radius / stellar_radius
    py_stellar_units = data['py'] * sun_radius / stellar_radius

    # If desired, show dashed circle for minimum distance
    if show_burn_circle:
        circle2 = plt.Circle(
            (0, 0),
            5,
            color='black',
            fill=False,
            linestyle='dashed',
            linewidth=0.5)  # dashed version
        # shaded version
        # circle2=plt.Circle((0,0), 5, color=(0.9, 0.9, 0.9), fill=True)
        fig.gca().add_artist(circle2)

    # Star in the center with limb darkening
    star_quality = 50  # number of shades
    limb1 = 0.4703  # quadratic limb darkening parameters
    limb2 = 0.236
    time_start = 0
    cNorm = matplotlib.colors.Normalize(vmin=-11.5, vmax=2)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='Reds_r')

    # Calculate and print deflection angle (and star name) if desired
    it = numpy.nditer(data['time'], flags=['f_index'])
    while not it.finished:

        x_location = data['px'][it.index] * sun_radius / stellar_radius
        y_location = data['py'][it.index] * sun_radius / stellar_radius
        speed = data['ship_speed'][it.index]
        time = data['time'][it.index]

        # Check if inside the plotted figure (faster plot generation)
        if abs(x_location) < scale and abs(y_location) < scale + 2:
            if time_start == 0:
                time_start = data['time'][it.index]
            plt.cla()

            # Colorbar
            last_label = 18.5
            for step in range(it.index):
                # Color bar
                x1 = 15
                x2 = 20
                y1 = data['py'][step] * sun_radius / stellar_radius
                y2 = y1
                if -20 < y1 < 20:
                    speed_change = data['ship_speed'][step] - data['ship_speed'][step-1]  # km/sec
                    time_change = data['time'][step] - data['time'][step-1]  # sec
                    g_force = speed_change / time_change * 1000 / 9.81
                    colorVal = scalarMap.to_rgba(g_force)
                    plt.plot([x1, x2], [y1, y2], color=colorVal, linewidth=4)
                if y1 < last_label + 1 and y1 < 18 and y1 > -18:
                    if g_force < -10:
                        textcolor = 'white'
                    else:
                        textcolor = 'black'
                    text = r'${:2.1f} g$'.format(g_force)
                    plt.annotate(
                        text,
                        xy=(17, y1),
                        fontsize=12,
                        horizontalalignment='center',
                        color=textcolor)

                    last_label = last_label - 4

            # Flight line
            plt.plot(
                px_stellar_units[:it.index + 1],
                py_stellar_units[:it.index + 1],
                color=flight_color,
                linewidth=0.5)

            # Star
            for i in range(star_quality):
                impact = i / float(star_quality)
                LimbDarkening = quadratic_limb_darkening(impact, limb1, limb2)
                Sun = plt.Circle(
                    (0, 0),
                    1 - i / float(star_quality),
                    color=(LimbDarkening, redness * LimbDarkening, 0))
                plt.gcf().gca().add_artist(Sun)

            ax.annotate(star_name, xy=(-2.5, 1.75))
            deflection_angle = abs(arctan2(data['py'][-1], data['px'][-1]) * (360 / (2 * pi))) - 90

            # print angle
            text_deflection_angle = r'$\delta = {:1.0f} ^\circ$'.format(deflection_angle)
            ax.annotate(
                text_deflection_angle,
                xy=(-scale + 1, scale - 2.5),
                fontsize=16)

            # Add a circle mark at the closest encounter
            index_min = numpy.argmin(data['stellar_distance'])
            if it.index > index_min:  # past the closest encounter
                stellar_radii = sun_radius / stellar_radius
                min_x_location = data['px'][index_min] * stellar_radii
                min_y_location = data['py'][index_min] * stellar_radii
                # print('drawing closest encounter at (x,y)=', min_x_location, min_y_location)
                marker = plt.Circle(
                    (min_x_location, min_y_location),
                    0.2,
                    color='red',
                    fill=False)
                plt.gcf().gca().add_artist(marker)

            # print weight ratio
            text_weight_ratio = r'$\sigma = {:1.1f}$ g/m$^2$'.format(weight_ratio)
            ax.annotate(
                text_weight_ratio,
                xy=(-scale + 1, scale - 5),
                fontsize=16)

            # print entry speed
            text_entry_speed = r'$\nu_{{\infty}} = {:10.0f}$ km/s'.format(data['ship_speed'][1])
            ax.annotate(
                text_entry_speed,
                xy=(-scale + 1, scale - 8),
                fontsize=16)

            # Add circle marks every [circle_spacing_minutes] and annotate them
            current_marker_number = 0

            # Yes, mark it
            marker = plt.Circle(
                (x_location, y_location), 0.1, color='black', fill=True)
            plt.gcf().gca().add_artist(marker)
            current_marker_number = current_marker_number + 1

            # Show sail angle as line
            angle = data['alpha'][it.index]
            if y_location > 0:
                angle = -angle

            # Sail is parallel, put it in direction of star:
            if not data['sail_not_parallel'][it.index]:
                v1_theta = arctan2(0, 0)
                v2_theta = arctan2(y_location, x_location)
                angle = (v2_theta - v1_theta) * (180.0 / 3.141)
                angle = radians(angle)

            # Calculate start and end positions of sail line
            length = 1
            endy = y_location + (length * sin(angle))
            endx = x_location + (length * cos(angle))
            starty = y_location - (length * sin(angle))
            startx = x_location - (length * cos(angle))
            plt.plot([startx, endx], [starty, endy], color='black')

            """
            # Add circle marks every [circle_spacing_minutes] and annotate them
            current_marker_number = 0
            inner_loop = numpy.nditer(data['time'], flags=['f_index'])
            while not inner_loop.finished and inner_loop.index < it.index:
                if (inner_loop[0] / 60) % circle_spacing_minutes == 0:
                    x_location = data['px'][inner_loop.index] * sun_radius / stellar_radius
                    y_location = data['py'][inner_loop.index] * sun_radius / stellar_radius
                    speed = data['ship_speed'][inner_loop.index]

                    # Check if inside the plotted figure (faster plot generation)
                    if abs(x_location) < scale and abs(y_location) < scale:

                        # Yes, mark it
                        marker = plt.Circle(
                            (x_location, y_location),
                            0.1,
                            color='black',
                            fill=True)
                        plt.gcf().gca().add_artist(marker)
                        current_marker_number = current_marker_number + 1

                        # To avoid crowding of text labels, set them sparsely
                        if data['ship_speed'][inner_loop.index] > 300:
                            n_th_print = 1  # print all labels
                        if 150 < data['ship_speed'][inner_loop.index] < 300:
                            n_th_print = 3  # print every 5th label
                        if data['ship_speed'][inner_loop.index] < 150:
                            n_th_print = 5  # print every 5th label

                        if -19 < y_location < 19 and (inner_loop[0] / 60) % (circle_spacing_minutes * n_th_print) == 0:
                            test_speed = ''
                            text_speed = r'{:10.0f} km/s'.format(speed)
                            ax.annotate(
                                text_speed,
                                xy=(x_location + 1, y_location - 0.5),
                                fontsize=12)
                inner_loop.iternext()
            """

            # Annotate current speed and time in lower left corner
            text_speed = r'Speed: {:10.0f} km/s'.format(speed)
            text_time = r'Time: {:10.0f} hrs'.format((time - time_start) / 60 / 60)
            ax.annotate(text_speed, xy=(-19, -17), fontsize=12)
            ax.annotate(text_time, xy=(-19, -19), fontsize=12)

            # Format the figure
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            minor_ticks = numpy.arange(-scale, scale, 1)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(minor_ticks, minor=True)
            plt.xlim(-scale, scale)
            plt.ylim(-scale, scale)
            plt.xlabel('Distance [Stellar Radii]', fontweight='bold')
            plt.ylabel('Distance [Stellar Radii]', fontweight='bold')
            fig.savefig(str(it.index) + ".png", bbox_inches='tight', dpi=400)
            print('it.index', it.index)

        it.iternext()

    return plt




def make_video_flight_black(
        data,
        stellar_radius,
        scale,
        flight_color,
        redness,
        show_burn_circle,
        star_name,
        weight_ratio,
        circle_spacing_minutes,
        annotate_cases,
        caption):
    fig = plt.gcf()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_axis_bgcolor('none')
    ax.set_axis_bgcolor((0, 0, 0))
    ax.patch.set_facecolor('black')
    fig.patch.set_facecolor('black')
    plt.axis('off')

    # Flight trajectorie
    px_stellar_units = data['px'] * sun_radius / stellar_radius
    py_stellar_units = data['py'] * sun_radius / stellar_radius

    # If desired, show dashed circle for minimum distance
    if show_burn_circle:
        circle2 = plt.Circle(
            (0, 0),
            5,
            color='black',
            fill=False,
            linestyle='dashed',
            linewidth=0.5)  # dashed version
        # shaded version
        # circle2=plt.Circle((0,0), 5, color=(0.9, 0.9, 0.9), fill=True)
        fig.gca().add_artist(circle2)

    # Star in the center with limb darkening
    star_quality = 50  # number of shades
    limb1 = 0.4703  # quadratic limb darkening parameters
    limb2 = 0.236
    time_start = 0
    cNorm = matplotlib.colors.Normalize(vmin=-11.5, vmax=2)
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap='Reds_r')

    # Calculate and print deflection angle (and star name) if desired
    it = numpy.nditer(data['time'], flags=['f_index'])
    while not it.finished:
        """
        fig = plt.gcf()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_axis_bgcolor('none')
        ax.set_axis_bgcolor((0, 0, 0))
        ax.patch.set_facecolor('black')
        fig.patch.set_facecolor('black')
        plt.axis('off')
        ax.set_axis_off()
        """
        x_location = data['px'][it.index] * sun_radius / stellar_radius
        y_location = data['py'][it.index] * sun_radius / stellar_radius
        speed = data['ship_speed'][it.index]
        time = data['time'][it.index]

        # Check if inside the plotted figure (faster plot generation)
        if abs(x_location) < scale and abs(y_location) < scale + 2:
            if time_start == 0:
                time_start = data['time'][it.index]
            plt.cla()

            # Colorbar
            last_label = 18.5
            for step in range(it.index):
                # Color bar
                x1 = 15
                x2 = 20
                y1 = data['py'][step] * sun_radius / stellar_radius
                y2 = y1
                if -20 < y1 < 20:
                    speed_change = data['ship_speed'][step] - data['ship_speed'][step-1]  # km/sec
                    time_change = data['time'][step] - data['time'][step-1]  # sec
                    g_force = speed_change / time_change * 1000 / 9.81
                    colorVal = scalarMap.to_rgba(g_force)
                    plt.plot([x1, x2], [y1, y2], color=colorVal, linewidth=4)
                if y1 < last_label + 1 and y1 < 18 and y1 > -18:
                    if g_force < -10:
                        textcolor = 'white'
                    else:
                        textcolor = 'black'
                    text = r'${:2.1f} g$'.format(g_force)
                    plt.annotate(
                        text,
                        xy=(17, y1),
                        fontsize=12,
                        horizontalalignment='center',
                        color=textcolor)

                    last_label = last_label - 4

            # Flight line
            plt.plot(
                px_stellar_units[:it.index + 1],
                py_stellar_units[:it.index + 1],
                color=flight_color,
                linewidth=0.5)

            # Star
            for i in range(star_quality):
                impact = i / float(star_quality)
                LimbDarkening = quadratic_limb_darkening(impact, limb1, limb2)
                Sun = plt.Circle(
                    (0, 0),
                    1 - i / float(star_quality),
                    color=(LimbDarkening, redness * LimbDarkening, 0))
                plt.gcf().gca().add_artist(Sun)

            ax.annotate(star_name, xy=(-2.5, 1.75))
            deflection_angle = abs(arctan2(data['py'][-1], data['px'][-1]) * (360 / (2 * pi))) - 90

            # print angle
            text_deflection_angle = r'$\delta = {:1.0f} ^\circ$'.format(deflection_angle)
            ax.annotate(
                text_deflection_angle,
                xy=(-scale + 1, scale - 2.5),
                fontsize=16,
                color='white')

            # Add a circle mark at the closest encounter
            index_min = numpy.argmin(data['stellar_distance'])
            if it.index > index_min:  # past the closest encounter
                stellar_radii = sun_radius / stellar_radius
                min_x_location = data['px'][index_min] * stellar_radii
                min_y_location = data['py'][index_min] * stellar_radii
                # print('drawing closest encounter at (x,y)=', min_x_location, min_y_location)
                marker = plt.Circle(
                    (min_x_location, min_y_location),
                    0.2,
                    color='red',
                    fill=False)
                plt.gcf().gca().add_artist(marker)

            # print weight ratio
            text_weight_ratio = r'$\sigma = {:1.1f}$ g/m$^2$'.format(weight_ratio)
            ax.annotate(
                text_weight_ratio,
                xy=(-scale + 1, scale - 5),
                fontsize=16,
                color='white')

            # print entry speed
            text_entry_speed = r'$\nu_{{\infty}} = {:10.0f}$ km/s'.format(data['ship_speed'][1])
            ax.annotate(
                text_entry_speed,
                xy=(-scale + 1, scale - 8),
                fontsize=16,
                color='white')

            """
            # Add circle marks every [circle_spacing_minutes] and annotate them
            current_marker_number = 0

            # Yes, mark it
            marker = plt.Circle(
                (x_location, y_location), 0.1, color='white', fill=True)
            plt.gcf().gca().add_artist(marker)
            current_marker_number = current_marker_number + 1
            """

            # Show sail angle as line
            angle = data['alpha'][it.index]
            if y_location > 0:
                angle = -angle

            # Sail is parallel, put it in direction of star:
            if not data['sail_not_parallel'][it.index]:
                v1_theta = arctan2(0, 0)
                v2_theta = arctan2(y_location, x_location)
                angle = (v2_theta - v1_theta) * (180.0 / 3.141)
                angle = radians(angle)

            # Calculate start and end positions of sail line
            length = 1
            endy = y_location + (length * sin(angle))
            endx = x_location + (length * cos(angle))
            starty = y_location - (length * sin(angle))
            startx = x_location - (length * cos(angle))
            plt.plot([startx, endx], [starty, endy], color='white')

            """
            # Add circle marks every [circle_spacing_minutes] and annotate them
            current_marker_number = 0
            inner_loop = numpy.nditer(data['time'], flags=['f_index'])
            while not inner_loop.finished and inner_loop.index < it.index:
                if (inner_loop[0] / 60) % circle_spacing_minutes == 0:
                    x_location = data['px'][inner_loop.index] * sun_radius / stellar_radius
                    y_location = data['py'][inner_loop.index] * sun_radius / stellar_radius
                    speed = data['ship_speed'][inner_loop.index]

                    # Check if inside the plotted figure (faster plot generation)
                    if abs(x_location) < scale and abs(y_location) < scale:

                        # Yes, mark it
                        marker = plt.Circle(
                            (x_location, y_location),
                            0.1,
                            color='black',
                            fill=True)
                        plt.gcf().gca().add_artist(marker)
                        current_marker_number = current_marker_number + 1

                        # To avoid crowding of text labels, set them sparsely
                        if data['ship_speed'][inner_loop.index] > 300:
                            n_th_print = 1  # print all labels
                        if 150 < data['ship_speed'][inner_loop.index] < 300:
                            n_th_print = 3  # print every 5th label
                        if data['ship_speed'][inner_loop.index] < 150:
                            n_th_print = 5  # print every 5th label

                        if -19 < y_location < 19 and (inner_loop[0] / 60) % (circle_spacing_minutes * n_th_print) == 0:
                            test_speed = ''
                            text_speed = r'{:10.0f} km/s'.format(speed)
                            ax.annotate(
                                text_speed,
                                xy=(x_location + 1, y_location - 0.5),
                                fontsize=12)
                inner_loop.iternext()
            """

            # Annotate current speed and time in lower left corner
            text_speed = r'Speed: {:10.0f} km/s'.format(speed)
            text_time = r'Time: {:10.0f} hrs'.format((time - time_start) / 60 / 60)
            ax.annotate(text_speed, xy=(-19, -17), fontsize=12, color='white')
            ax.annotate(text_time, xy=(-19, -19), fontsize=12, color='white')

            # Format the figure
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            #minor_ticks = numpy.arange(-scale, scale, 1)
            #ax.set_xticks(minor_ticks, minor=True)
            #ax.set_yticks(minor_ticks, minor=True)
            plt.xlim(-scale, scale)
            plt.ylim(-scale, scale)
            #plt.xlabel('Distance [Stellar Radii]', fontweight='bold')
            #plt.ylabel('Distance [Stellar Radii]', fontweight='bold')
            ax.set_axis_bgcolor('none')
            #ax.set_axis_bgcolor((0, 0, 0))

            plt.axis('off')
            ax.set_axis_off()
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)
            ax.patch.set_facecolor('black')
            fig.patch.set_facecolor('black')
            #circ=
            ax.add_patch(plt.Circle((0,0), radius=100, color='black', fill=True))
            fig.savefig(str(it.index) + ".png", bbox_inches='tight', dpi=400, pad_inches = 0,  facecolor=fig.get_facecolor())
            print('it.index', it.index)

        it.iternext()

    return plt
