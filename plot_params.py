# imports
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.time import Time
import os
import numpy as np
import math as m
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from cycler import cycler
import matplotlib.gridspec as gridspec # GRIDSPEC
from scipy.optimize import curve_fit
from math import cos, pi,sqrt
from astropy.constants import G, c, M_sun
from astropy import units as u
from scipy.stats import spearmanr
from scipy.odr import *
import argparse


def read_component_file(file):
    if "diskpbb" in file:
        return read_comp_diskpbb(file)
    elif "diskbb" or "bbody" or "powerlaw" in file:
        return read_comp_diskbb(file)
    elif "simpl" in file:
        return read_comp_simpl(file)
    elif "compTT" in file:
        return read_comp_compTT(file)
    else:
        return "error: unknown component file %s" % file


def read_comp_diskbb(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "<f8", "<f8", "<f8",
                                                                       "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                         converters={5: linked_to_errors, 6: linked_to_errors, 8: linked_to_errors, 9: linked_to_errors})

    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_comp_diskpbb(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "<f8", "<f8", "<f8",
                                                                       "<f8", "<f8", "<f8",
                                                                       "<f8", "<f8", "<f8"),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                         converters={5: linked_to_errors, 6: linked_to_errors, 8: linked_to_errors, 9: linked_to_errors,
                         11: linked_to_errors, 12: linked_to_errors})

    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_params_diskbb(comp_file):
    """Read diskbb file from fit_source_nofluxes.tcl
    Returns the data sorted by epoch"""
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "<f8", "<f8", "<f8",
                                                                       "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                         converters={5: linked_to_errors, 6: linked_to_errors, 8: linked_to_errors, 9: linked_to_errors})

    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_flux_diskbb(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "<f8", "<f8", "<f8"),
                         converters={4: logtoflux, 5: logtoflux_bounds, 6: logtoflux_bounds})

    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_pulsations_file(file):
    data = np.genfromtxt(file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "i8", "f8", "f8"),
                         missing_values={4: "", 5: "", 6:""}, filling_values={4:-1, 5:-1},
                         converters={})
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def pulsations_to_color(pulsations, pulse_fractions):
    color_list = []
    for pulsation, pulse_fraction in zip(pulsations, pulse_fractions):
        if int(pulsation)== -1:
            color = "white"
        elif int(pulsation)== 0:
            color = "red"
        elif int(pulsation)== 1:
            color = "green"
        else:
            color = "white"
        color_list.append(color)
    return np.array(color_list)


def pulse_fraction_to_color(pulse_fraction, pulsations):
    if float(pulsations)==-1:
        return "white"
    elif int(pulsations)==0:
        return "red"
        pulse_fraction * 0.2
    elif int(pulsations)==1:
        return "green"
    cmap = cm.get_cmap('Wistia')
    color = cmap(float(pulse_fraction)/40)
    return color


def pulsation_to_color(pulsation):
    if pulsation == 0:
        return "red"
    elif pulsation ==1:
        return "green"
    elif pulsation == -1:
        return "white"
    else:
        return "white"


def read_tbabs(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U11", "U14", "U10",
                                                                       "<f8", "<f8", "<f8"),
                         converters={5: linked_to_errors, 6: linked_to_errors})
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def linked_to_errors(x):
    # frozen parameter
    if float(x) == -3:
        print("Frozen parameter")
        return 0.000000001
    # unbound parameter
    elif float(x) == -5:
        print("Limit found")
        return 0
    # linked parameter
    elif float(x) == -4:
        print("Parameter linked")
        return 0.000000000000001
    else:
        return x


def read_comp_cutoffpl(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U14", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8","<f8", "<f8", "<f8",
                               "<f8", "<f8", "<f8"),
                         converters={13: logtoflux, 14: logtoflux_bounds,
                                     15: logtoflux_bounds})
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_comp_compTT(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U14", "<f8", "<f8", "<f8",
                          "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                          "<f8", "<f8", "<f8", "<f8", "<f8", "<f8","<f8", "<f8", "<f8"),
                         converters={22: logtoflux, 23: logtoflux_bounds,
                                     24: logtoflux_bounds})
    if len(np.atleast_1d(data))==1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_comp_simpl(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U14", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         converters={5: linked_to_errors, 6: linked_to_errors, 8: linked_to_errors,
                                     9: linked_to_errors, 13: logtoflux, 14: logtoflux_bounds, 15:logtoflux_bounds}, missing_values="not_computed")
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def tojd(x):
    return Time(x).jd


def logtoflux_bounds(x):
    # keep upper and lower bounds to 0
    if float(x) == 0:
        print("Flux bound found")
        return 0
    elif float(x) == -1:
        print("Warning; error in flux computation detected")
        return 0
    else:
        return m.pow(10, float(x))


def logtoflux(x):
    if float(x)==-1:
        print("Warning; error in flux computation detected")
        return 0
    return m.pow(10, float(x))


def read_fit_goodness(file):
    data = np.genfromtxt(file, delimiter='\t', names=True,
                         dtype=("U23", "U23", "U14", "U12", "U5", "f8", "i8", "f8"))
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order="epoch")
    return data


def read_flux_data(flux_file):
    data = np.genfromtxt(flux_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U12", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         missing_values='Error',
                         converters={4: logtoflux, 5: logtoflux_bounds, 6: logtoflux_bounds,
                                    7: logtoflux, 8: logtoflux_bounds,
                                   9: logtoflux_bounds, 10: logtoflux, 11: logtoflux_bounds, 12: logtoflux_bounds,
                                    13: logtoflux, 14: logtoflux_bounds, 15: logtoflux_bounds,
                                    16: logtoflux, 17: logtoflux_bounds, 18: logtoflux_bounds,
                                    19: logtoflux, 20: logtoflux_bounds, 21: logtoflux_bounds,
                                    22: logtoflux, 23: logtoflux_bounds, 24: logtoflux_bounds},
                         filling_values=0)
    if len(np.atleast_1d(data))==1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def remove_legend_repetitions(ax, fontsize=20):
    """Removes any repetead entries in the legend. Adds the legend to the plot too.
    Parameters
    ----------
    ax : The axis were the repeated entries are to be removed. """
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', prop={'size': fontsize})


def bounds_to_errors(values, lowerbounds, upperbounds):
    ''' Compute errors given the lower and upper bounds of a an array of values.
    Parameters:
    -----------
    value : the central values given by the fit
    lowerbound : its lower bounds
    upperbound : its upper bounds'''

    lower_errors = values - lowerbounds
    upper_errors = upperbounds - values

    for value, lowerbound, upperbound in zip(values, lowerbounds, upperbounds):
        if upperbound < value and upperbound != 0:
            print("Warning upperbound is lower than value!!! %.5f < %.5f" % (upperbound, value))
        if lowerbound > value and lowerbound != 0:
            print("Warning lowerbound is higher than value!!! %.5f > %.5f" % (lowerbound, value))

    uplims = np.zeros(values.shape)

    # lower bound (upper limit)
    uplims[np.where(lowerbounds == 0)] = 1
    lower_errors[np.where(lowerbounds == 0)] = (upperbounds[np.where(lowerbounds == 0)] - values[np.where(lowerbounds == 0)]) * 0.25
    values[np.where(lowerbounds == 0)] = upperbounds[np.where(lowerbounds == 0)]
    upper_errors[np.where(lowerbounds == 0)] = 0

    # upper bound found (lower limit)
    lolims = np.zeros(values.shape)
    lolims[np.where(upperbounds == 0)] = 1
    upper_errors[np.where(upperbounds == 0)] = (values[np.where(upperbounds == 0)] - lowerbounds[np.where(upperbounds == 0)]) * 0.25
    values[np.where(upperbounds == 0)] = lowerbounds[np.where(upperbounds == 0)]
    lower_errors[np.where(upperbounds == 0)] = 0
    return lower_errors, upper_errors, lolims, uplims


def jd_to_daymonthyear(x, pos):
    '''Format the axis to convert from Julian day to real date.'''
    time = Time(x, format='jd')
    time.format = 'iso'
    time.out_subfmt = 'date'
    return time


def compute_ratios(hard_flux, hard_flux_low, hard_flux_high, soft_flux, soft_flux_low, soft_flux_high):
    """Compute hardness ratios given fluxes in two energy bands."""
    ratio = hard_flux / soft_flux
    soft_err_low = soft_flux - soft_flux_low
    soft_err_high = soft_flux_high - soft_flux
    hard_err_low = hard_flux - hard_flux_low
    hard_err_high = hard_flux_high - hard_flux
    ratio_err_low = ((hard_err_low / soft_flux)**2 + (hard_flux * soft_err_low / soft_flux**2)**2)**(1 / 2)
    ratio_err_high = ((hard_err_high / soft_flux)**2 + (hard_flux * soft_err_high / soft_flux**2)**2)**(1 / 2)
    return ratio, ratio_err_low, ratio_err_high


def create_color_array(data_length, cmap='hsv'):
    """Create an array of colors given the length of a dataset. Useful for plots where a unique color is needed for each dataset.

    The returned colors come from the input map (jet by default).

    Parameters
    ----------
    data_length : The length of your data for the color array creation.

    """
    print("Creating color array for %i datasets" % data_length)
    x = np.arange(data_length)
    ys = [i + x + (i * x)**2 for i in range(data_length)]
    setmap = plt.get_cmap(name=cmap)

    colors = setmap(np.linspace(0, 1, len(ys)))
    return colors


def get_markers_array(data_length):
    """Get an array of markers given the length of a dataset. Useful for plots where a unique marker is needed for each dataset.

    There are 17 different markers and after that they are repeated.

    Parameters
    ----------
    data_length : The length of your data for the marker array creation.

    """
    m = ['o', '^', (12, 1, 50), "s", 'v', 'p', 'P',  'd', '*', 'h', (5, 1, 10), (3, 0, 10), 'D', '8' , (10, 1, 20),
         '<', (12, 1, 120), '.', '>', (7, 0, 30), (20, 0, 50), (20, 0, 34), '1']

    while data_length > len(m):
        m.extend(m)

    return m


def draw_arrows(x, y, colors, ax=None):
    for i in np.arange(1, len(x)):
        if ax==None:
            plt.annotate("", xy=(x[i-1],y[i-1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="<-", shrinkA=10, shrinkB=10, color=colors[i-1]))
        else:
            ax.annotate("", xy=(x[i-1],y[i-1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="<-", shrinkA=10, shrinkB=10, color=colors[i-1]))


def fit_diskbb(fluxes, temperatures, temperatures_errlow, temperatures_errhigh, b=4):
    popt, pconv = curve_fit(diskLvsT, temperatures, fluxes,
                          sigma=(temperatures_errlow + temperatures_errhigh)/2,
                          p0=[10**-13,b])
    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pconv))
    return a, b, a_err, b_err

    return


def fit_outflow(luminosity, temperatures, luminosity_errlow, luminosity_errhigh):
    popt, pconv = curve_fit(diskLvsT, temperatures, luminosity,
                          sigma=(luminosity_errlow + luminosity_errhigh)/ 2,
                          p0=[1.74,-1])
    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pconv))
    return a, b, a_err, b_err


def diskLvsT(x, a, b):
    return a * x**b


def radius_tomass(r_in):
    return r_in * (c.to("km/s")) ** 2 / 2 / G.to("km**3/kg/s**2") / M_sun


def disknorm_tomass(norm, distance, angle=60):
    r_in = disknorm_tosize(norm, distance, angle)
    return radius_tomass(r_in)


def disknorm_tosize(norm, distance, angle=60):
    """Returns the disk radius in km
    distance: distance to the source in megaparsecs"""
    angle_rad = 60 / 360 * 2 * pi
    # mega parsecs to kiloparsecs
    distance_kpc = distance * 10 ** 3
    r_in = np.sqrt(norm / cos(angle_rad)) * (distance_kpc / 10) * u.km
    return r_in


def plotdisk(data, ax, param_1="Tin", param_2="flux"):
    if len(np.atleast_1d(data)) > 1:
        colors = create_color_array(len(data["epoch"]), "jet")
    else:
        colors = ["cyan"]
    ax.set_prop_cycle(cycler('color', colors))
    param_1err_low, param_1err_high, param_1lolimits, param_1uplimits = bounds_to_errors(data["%s" %param_1],
                                                                                         data["%slow" %param_1],
                                                                                         data["%supp" %param_1])
    param_2err_low, param_2err_high, param_2lolimits, param_2uplimits = bounds_to_errors(data["%s" %param_2],
                                                                                         data["%slow" %param_2],
                                                                                         data["%supp" %param_2])
    for index in np.arange(0, len(data["epoch"])):
        if data["xmm_obsid"][index] != "":
            label = data["xmm_obsid"][index]
        else:
            label = data["chandra"][index]

        ax.errorbar(data["%s" % param_1][index], data["%s" % param_2][index],
                             xerr=[[param_1err_low[index]], [param_1err_high[index]]],
                             yerr=[[param_2err_low[index]], [param_2err_high[index]]], label=label,
                             marker="$ f $", xlolims=param_1lolimits[index], xuplims=param_1uplimits[index],
                             uplims=param_2uplimits[index], lolims=param_2lolimits[index], markersize=10)
def eddington_limit(M):
    return 1.26 * M * 10**38 / 10**39


def bolometric_l(M, m_dot):
    return eddington_limit(M) * (1 + 3/5 * np.log(m_dot))


def t_disk_max(M, m_dot):
    return 1.6 * (M)**(-1/4) * (1 - 0.2 * m_dot**(-1/3))
def t_spherization_max(M, m_dot):
    return 1.5 * (M)**(-1/4) * m_dot**(-1/2) * (1 + 0.3 * m_dot**(-3/4))
def t_photosphere_max(M, m_dot, beta=1, chi=1, epsilon_wind=1/2):
    return 0.8 * (beta * chi / epsilon_wind)**1/2 * M**(-1/4) * m_dot**(-3/4)


def readbroadbandfile(broadband_file="~/x_ray_data/broadband_fitting_plot.config"):
    plot_config = np.genfromtxt(broadband_file, delimiter="\t\t", dtype=("U13", "U36", "U18", float, float), names=True)
    if len(np.atleast_1d(plot_config))== 1:
        plot_config = np.array([plot_config])
    return plot_config


plt.style.use('~/.config/matplotlib/stylelib/paper.mplstyle')

# read arguments
ap = argparse.ArgumentParser(description='Spectrum fits to be loaded')
ap.add_argument("source", nargs='+', help="Source whose parameters are to be plotted")
ap.add_argument("-c", '--chandra', nargs='?', help="Include chandra data on the plots", default=False)
ap.add_argument("-m", '--model_dir', nargs='?', help="Model directory with the result of the fit", default=os.path.split(os.getcwd())[1], type=str)
args = ap.parse_args()

# simbad radius in arseconds
plot_source = args.source[0]
include_chandra = args.chandra

model_dir = args.model_dir
tbabs_figure, tbabs_ax = plt.subplots(1, 1)

print("Model directory %s" % model_dir)

param_1 = 'nH'
param_2 = "epoch"

common_dir = "%s/x_ray_data/" % os.getenv("HOME")
source_dir = "%s/%s/%s" % (common_dir, plot_source, model_dir)
print("Processing source %s" % plot_source)

# get source distance
f = open("%s/%s/source_params.txt" % (common_dir, plot_source), "r")
lines = f.readlines()
f.close()
megaparsecs = float(lines[1])
source_name = lines[6]

print('Source distance %.1f Mpc' % megaparsecs)

tbabs_figure, tbabs_ax = plt.subplots(1, 1)

if os.path.isfile("%s/components/tbabs_1.dat" % source_dir):
    data = read_tbabs("%s/components/tbabs_1.dat" % source_dir)
    total_data = data

    if os.path.isfile("%s/components/chandra_tbabs_1.dat" % source_dir) and include_chandra:
        chandra_data = read_tbabs("%s/components/chandra_tbabs_1.dat" % source_dir)
        print("Found Chandra %d observations" % len(np.atleast_1d(chandra_data)))
        if len(np.atleast_1d(chandra_data)) == 1:
            chandra_data = np.array(chandra_data)
            chandra_data.sort(order="epoch")
        total_data = np.append(total_data, chandra_data)
        param_1err_low, param_1err_high, param_1lolimits, param_1uplimits = bounds_to_errors(chandra_data["%s" % param_1],
                                                                                             chandra_data["%slow" % param_1],
                                                                                             chandra_data["%supp" % param_1])

        tbabs_ax.errorbar(Time(chandra_data["epoch"]).jd, chandra_data["%s" % param_1], yerr=[param_1err_low, param_1err_high], color="green",
                          uplims=param_1lolimits, lolims=param_1uplimits, fmt="s")

elif os.path.isfile("%s/components/chandra_tbabs_1.dat" % source_dir):
    print("XMM-Newton data not found")
    data = read_tbabs("%s/components/chandra_tbabs_1.dat" % source_dir)
    print("Found Chandra %d observations" % len(data))

data.sort(order='epoch')
print(total_data["%supp" % param_1])
print(total_data["%s" % param_1])
# skip upper and lower limits
data_nolimits = np.array([data["%s" % param_1] for data in total_data if not data["%slow" % param_1] == 0 and not data["%supp" % param_1] == 0])

median_param1 = np.median(data_nolimits)
mean_param1 = np.average(data_nolimits)

print("%s \n median value: %.3f \n mean value: %.3f" % (param_1, median_param1, mean_param1))

param_1err_low, param_1err_high, param_1lolimits, param_1uplimits = bounds_to_errors(data["%s" % param_1],
                                                                                     data["%slow" % param_1],
                                                                                     data["%supp" % param_1])


tbabs_ax.errorbar(Time(data["epoch"]).jd, data["%s" % param_1], yerr=[param_1err_low, param_1err_high], color="black",
                 uplims=param_1uplimits, lolims=param_1lolimits, fmt=".")
#tbabs_ax.axhline(y=median_param1, color='navy', ls='--', label='median %.3f' % median_param1)
tbabs_ax.axhline(y=mean_param1, color='navy', ls='solid', label='mean %.3f' % mean_param1)

tbabs_ax.set_xlabel("Date")
tbabs_ax.set_ylabel("%s 10$^{22}$ cm$^{-2}$" % param_1)
date_formatter = FuncFormatter(jd_to_daymonthyear)
plt.xticks(rotation=45)
# save components flux plot
tbabs_ax.xaxis.set_major_formatter(date_formatter)
tbabs_ax.legend()
tbabs_figure.savefig("%s/tbabs_time.png" % source_dir, format="png")

param_2 = 'norm'
components = ["diskbb_0", "diskbb_1", "diskpbb_0", "bbody_0", "powerlaw_0"]
params_1 = ["Tin", "Tin", "Tin", "kT", "PhoIndex"]

for component, param_1 in zip(components, params_1):
    disks_figure, diskbb_T_ax = plt.subplots(1, 1)
    if not os.path.isfile("%s/components/%s.dat" % (source_dir, component)):
        print("Component %s not found " % component)
        continue
    data = read_component_file("%s/components/%s.dat" % (source_dir, component))
    print("Found XMM-Newton %d observations for component %s" % (len(data), component))

    if os.path.isfile("%s/components/chandra_%s.dat" % (source_dir, component)) and include_chandra:
        chandra_data = read_component_file("%s/components/chandra_%s.dat" % (source_dir, component))
        print("Found Chandra %d observations" % len(np.atleast_1d(chandra_data)))
        data = np.append(data, chandra_data)

    data.sort(order="epoch")
    colors = create_color_array(len(data["epoch"]), "plasma")
    markers = get_markers_array(len(data["epoch"]))

    diskbb_T_ax.set_prop_cycle(cycler('color', colors))
    param_1err_low, param_1err_high, param_1lolimits, param_1uplimits = bounds_to_errors(data["%s" % param_1],
                                                                                         data["%slow" % param_1],
                                                                                         data["%supp" % param_1])
    param_2err_low, param_2err_high, param_2lolimits, param_2uplimits = bounds_to_errors(data["%s" % param_2],
                                                                                         data["%slow" % param_2],
                                                                                         data["%supp" % param_2])
    for index in np.arange(0, len(data["epoch"])):
        if data["xmm_obsid"][index] != "":
            label = data["epoch"][index]
        else:
            label = data["epoch"][index]
        print("Plotting observation %s" % label)
        diskbb_T_ax.errorbar(data["%s" % param_1][index], data["%s" % param_2][index],
                             xerr=[[param_1err_low[index]], [param_1err_high[index]]],
                             yerr=[[param_2err_low[index]], [param_2err_high[index]]], label=label,
                             marker=markers[index], xlolims=param_1lolimits[index], xuplims=param_1uplimits[index],
                             uplims=param_2uplimits[index], lolims=param_2lolimits[index])

        #draw_arrows(data["%s" % param_1], data["%s" % param_2], colors, diskbb_T_ax)

    diskbb_T_ax.set_xlabel("%s (%s) keV" % (param_1, component))
    diskbb_T_ax.set_ylabel("%s" % param_2)
    diskbb_T_ax.legend()
    ax2 = diskbb_T_ax.twinx()
    ax2.set_ylabel('Radius (km)')
    y_ticks = diskbb_T_ax.get_yticks()
    y2labels = ['{0:.0f}'.format(radius) for radius in disknorm_tosize(y_ticks, megaparsecs)]
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y2labels)
    ax2.set_ylim(diskbb_T_ax.get_ylim())
    disks_figure.savefig("%s/%s_t_norm.png" % (source_dir, component), format="png")


gamma_figure, gamma_ax = plt.subplots(1, 1)
param_1 = 'Gamma'
param_2 = 'FracSctr'
if os.path.isfile("%s/components/simpl_0.dat" % source_dir):
    simpl_data = read_comp_simpl("%s/components/simpl_0.dat" % source_dir)
    plot_simpl = 1
else:
    plot_simpl = 0
if os.path.isfile("%s/components/chandra_simpl_0.dat" % source_dir) and include_chandra:
    chandra_data = read_comp_simpl("%s/components/chandra_simpl_0.dat" % (source_dir))
    print("Found Chandra %d observations" % len(np.atleast_1d(chandra_data)))
    simpl_data = np.append(simpl_data, chandra_data)

if plot_simpl == 1:
    simpl_data.sort(order="epoch")
    colors = create_color_array(len(simpl_data["epoch"]), "plasma")
    markers = get_markers_array(len(simpl_data["epoch"]))
    gamma_ax.set_prop_cycle(cycler('color', colors))
    print("%s: average frac: %.2f" % (source_dir, np.mean(simpl_data["%s" % param_2])))

    gamma_err_low, gamma_err_high, gamma_lolimits, gamma_uplimits = bounds_to_errors(simpl_data["%s" % param_1], simpl_data["%slow" % param_1],
                                                                                                 simpl_data["%supp" % param_1])

    param2_err_low, param2_err_high, param2_lolimits, param2_uplimits = bounds_to_errors(simpl_data["%s" % param_2],
                                                                                                 simpl_data["%slow" % param_2],
                                                                                                 simpl_data["%supp" % param_2])

    print("Found %d observations" % len(simpl_data))
    for index, epoch in enumerate(simpl_data["epoch"]):
        gamma_ax.errorbar(simpl_data["%s" % param_2][index], simpl_data["%s" % param_1][index],
                 yerr=[[gamma_err_low[index]], [gamma_err_high[index]]], label=epoch, marker=markers[index], xerr=[[param2_err_low[index]], [param2_err_high[index]]],
                     uplims=gamma_uplimits[index], lolims=gamma_lolimits[index], xlolims=param2_lolimits[index], xuplims=param2_uplimits[index])

    #draw_arrows(simpl_data["%s" % param_2], simpl_data["%s" % param_1], colors, gamma_ax)
    plt.xlabel("%s" % param_2)
    plt.ylabel("%s" % param_1)
    plt.legend()
    gamma_figure.savefig("%s/gamma.png" % (source_dir), format="png")

print("Creating goodness figure")
fit_goodness_figure, fit_goodness_ax = plt.subplots(1, 1)
fit_goodness_file = "%s/fit_goodness.log" % (source_dir)
fit_goodness = read_fit_goodness(fit_goodness_file)
bins = np.arange(0.65, 1.55, 0.1)
fit_goodness_ax.hist(x=fit_goodness["chisqr"], bins=bins, color="green", alpha=1.0, linewidth=3, edgecolor="black")
fit_goodness_ax.axvline(x=1, color="black", ls="--")
fit_goodness_ax.set_xlabel("$\chi_r$")
fit_goodness_ax.set_ylabel("N")
fit_goodness_figure.savefig("%s/goodness.png" % source_dir, format="png", tight_layout=True)
