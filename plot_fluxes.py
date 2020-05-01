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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker
from cycler import cycler
import matplotlib.gridspec as gridspec # GRIDSPEC
from scipy.optimize import curve_fit
from math import cos, pi, sqrt
from astropy.constants import G, c, M_sun
from astropy import units as u
from scipy.stats import spearmanr
import scipy.odr.odrpack as odr
import argparse
import sys


def read_component_file(file):
    if "diskbb" or "bbody" or "powerlaw" in file:
        return read_comp_diskbb(file)
    elif "simpl" in file:
        return read_comp_simpl(file)
    elif "compTT" in file:
        return read_comp_compTT(file)
    elif "diskpbb" in file:
        return read_comp_diskpbb(file)
    else:
        return "error: unknown component file %s" % file


def read_comp_diskpbb(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "<f8", "<f8", "<f8",
                                                                       "<f8", "<f8", "<f8",
                                                                       "<f8", "<f8", "<f8"),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                         converters={5: linked_to_errors, 6: linked_to_errors, 8: linked_to_errors, 9: linked_to_errors,
                         10: linked_to_errors, 11: linked_to_errors})

    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_component_params_file(file):
    if "diskbb" or "simpl" or "bbody" or "powerlaw" in file:
        return read_params_diskbb(file)
    elif "compTT" in file:
        return read_comp_compTT(file)
    elif "diskpbb" in file:
        return read_params_diskpbb(file)
    else:
        return "error: unknown component file %s" % file


def read_comp_diskbb(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "<f8", "<f8", "<f8",
                                                                       "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         converters={5: linked_to_errors, 6: linked_to_errors, 8: linked_to_errors, 9: linked_to_errors, 10: logtoflux, 11: logtoflux_bounds, 12: logtoflux_bounds})

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


def read_flux_comp(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "<f8", "<f8", "<f8"),
                         converters={4: logtoflux, 5: logtoflux_bounds, 6: logtoflux_bounds}, comments="#")

    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_pulsations_file(file):
    data = np.genfromtxt(file, delimiter='\t', names=True, dtype=("U23", "U13", "U14", "U11",
                                                                       "i8", "f8", "f8"),
                         missing_values={4: "", 5: "", 6: ""}, filling_values={4: -1, 5: -1},
                         converters={})
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def pulsations_to_color(pulsations, pulse_fractions):
    color_list = []
    for pulsation, pulse_fraction in zip(pulsations, pulse_fractions):
        if int(pulsation) == -1:
            color = "white"
        elif int(pulsation) == 0:
            color = "red"
        elif int(pulsation) == 1:
            color = "green"
        else:
            color = "white"
        color_list.append(color)
    return np.array(color_list)


def pulse_fraction_to_color(pulse_fraction, pulsations):
    if float(pulsations) == -1:
        return "white"
    elif int(pulsations) == 0:
        return "red"
        pulse_fraction * 0.2
    elif int(pulsations) == 1:
        return "green"
    cmap = cm.get_cmap('Wistia')
    color = cmap(float(pulse_fraction) / 40)
    return color


def pulsation_to_color(pulsation):
    if pulsation == 0:
        return "red"
    elif pulsation == 1:
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
        return 0.0000001
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
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
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
                          "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         converters={22: logtoflux, 23: logtoflux_bounds,
                                     24: logtoflux_bounds})
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_comp_simpl(comp_file):
    data = np.genfromtxt(comp_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U14", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         converters={5: linked_to_errors, 6: linked_to_errors, 8: linked_to_errors,
                                     9: linked_to_errors, 13: logtoflux, 14: logtoflux_bounds, 15: logtoflux_bounds})
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
    if float(x) == -1:
        print("Warning; error in flux computation detected")
        return 0
    return m.pow(10, float(x))


def read_fit_goodness(file):
    data = np.genfromtxt(file, delimiter='\t', names=True,
                         dtype=("U23", "U23", "U14", "U12", "f8", "i8", "f8", "f8"))
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order="epoch")
    return data


def read_flux_data(flux_file):
    data = np.genfromtxt(flux_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U12",
                                "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         missing_values='Error',
                         converters={4: logtoflux, 5: logtoflux_bounds, 6: logtoflux_bounds,
                                     7: logtoflux, 8: logtoflux_bounds,
                                     9: logtoflux_bounds, 10: logtoflux, 11: logtoflux_bounds, 12: logtoflux_bounds,
                                     13: logtoflux, 14: logtoflux_bounds, 15: logtoflux_bounds,
                                     16: logtoflux, 17: logtoflux_bounds, 18: logtoflux_bounds,
                                     19: logtoflux, 20: logtoflux_bounds, 21: logtoflux_bounds,
                                     22: logtoflux, 23: logtoflux_bounds, 24: logtoflux_bounds,
                                     25: logtoflux, 26: logtoflux_bounds, 27: logtoflux_bounds,
                                     28: logtoflux, 29: logtoflux_bounds, 30: logtoflux_bounds},
                         filling_values=0)
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_newflux_data(flux_file):
    """Read flux file produced with fit_source.tcl"""
    data = np.genfromtxt(flux_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U12", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8"),
                         missing_values='Error',
                         converters={4: logtoflux, 5: logtoflux_bounds, 6: logtoflux_bounds, 7: logtoflux, 8: logtoflux_bounds,
                         9: logtoflux_bounds, 10: logtoflux, 11: logtoflux_bounds, 12: logtoflux_bounds,
                         13: logtoflux, 14: logtoflux_bounds, 15: logtoflux_bounds,
                         16: logtoflux, 17: logtoflux_bounds, 18: logtoflux_bounds,
                         19: logtoflux, 20: logtoflux_bounds, 21: logtoflux_bounds,
                         22: logtoflux, 23: logtoflux_bounds, 24: logtoflux_bounds, 25: logtoflux, 26: logtoflux_bounds, 27: logtoflux_bounds,
                         28: logtoflux, 29: logtoflux_bounds, 30: logtoflux_bounds}, filling_values=0)

    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    data.sort(order='epoch')
    return data


def read_newflux_data_new(flux_file):
    """Read flux file produced with fit_source.tcl"""
    data = np.genfromtxt(flux_file, delimiter='\t', names=True,
                         dtype=("U23", "U13", "U7", "U12", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
                                "<f8", "<f8", "<f8"),
                         missing_values='Error',
                         converters={4: logtoflux, 5: logtoflux_bounds, 6: logtoflux_bounds, 7: logtoflux, 8: logtoflux_bounds,
                         9: logtoflux_bounds, 10: logtoflux, 11: logtoflux_bounds, 12: logtoflux_bounds,
                         13: logtoflux, 14: logtoflux_bounds, 15: logtoflux_bounds,
                         16: logtoflux, 17: logtoflux_bounds, 18: logtoflux_bounds,
                         19: logtoflux, 20: logtoflux_bounds, 21: logtoflux_bounds,
                         22: logtoflux, 23: logtoflux_bounds, 24: logtoflux_bounds,
                         }, filling_values=0)

    if len(np.atleast_1d(data)) == 1:
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
        if upperbound < value and upperbound!=0:
            print("Warning upperbound is lower than value!!! %.5f < %.5f" % (upperbound, value))
        if lowerbound > value and lowerbound!=0:
            print("Warning lowerbound is higher than value!!! %.5f > %.5f" % (lowerbound, value))

    uplims = np.zeros(values.shape)
    # lower bound (upper limit)
    uplims[np.where(lowerbounds==0)] = 1
    lower_errors[np.where(lowerbounds==0)] = (upperbounds[np.where(lowerbounds==0)] - values[np.where(lowerbounds==0)]) * 0.2
    values[np.where(lowerbounds==0)] = upperbounds[np.where(lowerbounds==0)]
    upper_errors[np.where(lowerbounds==0)] = 0

    # upper bound found (lower limit)
    lolims = np.zeros(values.shape)
    lolims[np.where(upperbounds==0)] = 1
    upper_errors[np.where(upperbounds==0)] = (values[np.where(upperbounds==0)] - lowerbounds[np.where(upperbounds==0)]) * 0.2
    values[np.where(upperbounds==0)] = lowerbounds[np.where(upperbounds==0)]
    lower_errors[np.where(upperbounds==0)] = 0
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
    m = ['o', '^', (12, 1, 50), "s", 'v', 'p', 'P', 'd', '*', 'h', (5, 1, 10), (3, 0, 10), 'D', '8', (10, 1, 20),
         '<', (12, 1, 120), '.', '>', (7, 0, 30), (20, 0, 50), (20, 0, 34), '1']

    while data_length > len(m):
        m.extend(m)
    return m


def draw_arrows(x, y, colors, ax=None):
    for i in np.arange(1, len(x)):
        if ax is None:
            plt.annotate("", xy=(x[i-1], y[i-1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="<-", shrinkA=10, shrinkB=10, color=colors[i-1]))
        else:
            ax.annotate("", xy=(x[i-1], y[i-1]), xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="<-", shrinkA=10, shrinkB=10, color=colors[i-1]))


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
                          sigma=(luminosity_errlow + luminosity_errhigh)/2,
                          p0=[1.74,-1])
    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pconv))
    return a, b, a_err, b_err


def fit_diskbb_odr(x, x_errlow, x_errhigh, y, y_errlow, y_errhigh, index=4, norm=5):
    """Fit disk temperature vs luminosity relationship.
    x: the temperature of the disk
    y: the luminosity of the disk"""
    model = odr.Model(diskLvsT_odr)
    data = odr.Data(x, y, wd=1. / ((x_errlow + x_errhigh) / 2)**2, we=1. / ((y_errlow + y_errhigh) / 2)**2)
    myodr = odr.ODR(data, model, beta0=[norm, index])
    output = myodr.run()
    return output.beta[0], output.beta[1], output.sd_beta[0], output.sd_beta[1]


def diskLvsT(x, a, b):
    return a * x**b


def diskLvsT_odr(B, x):
    return B[0] * x**B[1]


def radius_tomass(r_in):
    return r_in * (c.to("km/s")) ** 2 / 6 / G.to("km**3/kg/s**2") / M_sun


def disknorm_tomass(norm, distance, angle=60):
    r_in = disknorm_tosize(norm, distance, angle)
    return radius_tomass(r_in)


def disknorm_tosize(norm, distance, angle=60):
    """Returns the disk radius in km"""
    angle_rad = 60 / 360 * 2 * pi
    # mega parsecs to parsecs
    distance = distance * 10 ** 3
    r_in = np.sqrt(norm / cos(angle_rad)) * (distance / (10)) * u.km
    return r_in


def plotdisk(data, ax, param_1="Tin", param_2="flux"):
    if len(np.atleast_1d(data)) > 1:
        colors = create_color_array(len(data["epoch"]), "jet")
        markers = get_markers_array(len(data["epoch"]))
    else:
        colors = ["cyan"]
        markers = ['x']
    ax.set_prop_cycle(cycler('color', colors))
    param_1err_low, param_1err_high, param_1lolimits, param_1uplimits = bounds_to_errors(data["%s" %param_1],
                                                                                         data["%slow" %param_1],
                                                                                         data["%supp" %param_1])
    param_2err_low, param_2err_high, param_2lolimits, param_2uplimits = bounds_to_errors(data["%s" %param_2],
                                                                                         data["%slow" %param_2],
                                                                                         data["%supp" %param_2])
    for index in np.arange(0, len(data["epoch"])):
        if data["xmm_obsid"][index]!="":
            label=data["xmm_obsid"][index]
        else:
            label=data["chandra"][index]

        ax.errorbar(data["%s" %param_1][index], data["%s" %param_2][index] ,
                             xerr=[[param_1err_low[index]], [param_1err_high[index]]],
                             yerr=[[param_2err_low[index]], [param_2err_high[index]]], label=label,
                             marker="$ f $", xlolims=param_1lolimits[index], xuplims=param_1uplimits[index],
                             uplims=param_2uplimits[index], lolims=param_2lolimits[index])


def eddington_limit(M):
    return 1.26 * M * 10**38 / 10**39


def bolometric_l(M, m_dot):
    return eddington_limit(M) * (1 + 3/5 * np.log(m_dot))


def t_disk_max(M, m_dot):
    return 1.6 * (M)**(-1/4) * (1 - 0.2 * m_dot**(-1/3))


def t_spherization_max(M, m_dot):
    return 1.5 * (M)**(-1/4) * m_dot**(-1/2) * (1 + 0.3 * m_dot**(-3/4))


def t_photosphere_max(M, m_dot, beta=1, chi=1, epsilon_wind=1/2):
    return 0.8 * (beta * chi / epsilon_wind)**1/2 * M**(-1/4) * m_dot**(-3 / 4)


def readbroadbandfile(broadband_file="/home/agurpide/x_ray_data/broadband_fitting_plot.config"):
    plot_config = np.genfromtxt(broadband_file, delimiter="\t\t", dtype=("U13", "U36", "U18", float, float), names=True)
    if len(np.atleast_1d(plot_config)) == 1:
        plot_config = np.array([plot_config])
    return plot_config


plt.style.use('/home/agurpide/.config/matplotlib/stylelib/paper.mplstyle')

# read arguments
ap = argparse.ArgumentParser(description='Spectrum fits to be loaded')
ap.add_argument("source", nargs='+', help="Source whose parameters are to be plotted")
ap.add_argument("-m", '--model_dir', nargs='?', help="Model directory with the result of the fit", default=os.path.split(os.getcwd())[1], type=str)
args = ap.parse_args()

# simbad radius in arseconds
plot_sources = args.source

model_dir = args.model_dir
for plot_source in plot_sources:
    tbabs_figure, tbabs_ax = plt.subplots(1, 1)
    plot_config = readbroadbandfile()

    hard_bands = ["1030", "1510"]
    soft_bands = ["03100", "0315"]
    xlabels = ["F [10 $-$ 30 keV] / F [0.3 $-$ 10 keV]", "F [1.5 $-$ 10 keV] / F [0.3 $-$ 1.5 keV]"]
    total_band = "03100"

    common_dir = "/home/agurpide/x_ray_data/"
    source_dir = "%s/%s/%s" % (common_dir, plot_source, model_dir)
    flux_file = "%s/fluxes/fluxes.dat" % source_dir
    flux_color_file = "%s/fluxes/fluxes_color.dat" % source_dir

    # get source distance
    f = open("%s/%s/source_params.txt" % (common_dir, plot_source), "r")
    lines = f.readlines()
    f.close()
    megaparsecs = float(lines[1])
    source_name = lines[6]

    print('Source %s at distance %.1f Mpc' % (source_name, megaparsecs))
    parsecs = megaparsecs * m.pow(10, 6)
    distancecm = u.pc.to(u.cm, parsecs)
    constant = 4 * m.pi * distancecm ** 2 / 10**39

    # hardness ratio
    for soft_band, hard_band, xlabel in zip(soft_bands, hard_bands, xlabels):
        flux_figure, flux_ax = plt.subplots(1, 1)
        if not os.path.isfile("%s" % flux_file):
            print("Fluxes not found for this source")
            continue
        flux_data = read_newflux_data("%s" % flux_file)
        ratio_soft, ratio_err_low_soft, ratio_err_high_soft = compute_ratios(flux_data["%s" % hard_band],
                                                                             flux_data["%s_lower" % hard_band],
                                                                             flux_data["%s_upper" % hard_band], flux_data["%s" % soft_band],
                                                                             flux_data["%s_lower" % soft_band], flux_data["%s_upper" % soft_band])

        markers = get_markers_array(len(flux_data["epoch"]))
        colors = create_color_array(len(flux_data["epoch"]), "plasma")

        print("Found %d observations" % len(flux_data))

        for index, color in enumerate(colors):
            if flux_data["xmm_obsid"][index] != "":
                label = flux_data["epoch"][index]
                if flux_data["nustar_obsid"][index] != "":
                    facemarkercolor = "None"
                else:
                    facemarkercolor = color
            else:
                label = flux_data["epoch"][index]
                facemarkercolor = "black"
            flux_ax.errorbar(ratio_soft[index], constant * flux_data["%s" % total_band][index], xerr=[[ratio_err_low_soft[index]], [ratio_err_high_soft[index]]],
                             yerr=[[constant * (flux_data["%s" % total_band] - flux_data["%s_lower" % total_band])[index]],
                             [constant * (flux_data["%s_upper" % total_band] - flux_data["%s" % total_band])[index]]],
                             label=label, markerfacecolor=facemarkercolor, color=color, marker=markers[index], fmt="-")
        draw_arrows(ratio_soft, constant * flux_data["%s" % total_band], colors)
        f = open("%s/hardness_%s.txt" % (source_dir, soft_band), "w+")

        out_string_hardness = "%.2f\t%.3f\t%.3f\n" % (np.mean(ratio_soft), sum(ratio**2 for ratio in ratio_err_low_soft),
                                                          sum(ratio**2 for ratio in ratio_err_high_soft))

        f.write("#<hardness> \n %s" % (out_string_hardness))
        f.close()
        flux_ax.set_xlabel("%s" % xlabel)
        flux_ax.set_ylabel("L [0.3 $-$ 10 keV] (10$^{39}$ erg/s)")
        plt.tight_layout()
        flux_ax.legend()
        print("Saving flux hardness %s" % soft_band)
        flux_figure.savefig("%s/%sflux_hardness.png" % (source_dir, soft_band), bbox_inches='tight', format="png")

    if os.path.isfile("%s" % flux_color_file):
        print("Plotting color-color diagram")
        color_flux_data = read_newflux_data("%s" % flux_color_file)
        flux_figure, flux_ax = plt.subplots(1, 1)
        hard_band = "1030"
        soft_band = "2050"
        mid_band = "50100"
        ratio_hard, ratio_err_low_hard, ratio_err_high_hard = compute_ratios(color_flux_data["%s" % hard_band],
                                                                             color_flux_data["%s_lower" % hard_band],
                                                                             color_flux_data["%s_upper" % hard_band], color_flux_data["%s" % mid_band],
                                                                             color_flux_data["%s_lower" % mid_band], color_flux_data["%s_upper" % mid_band])
        ratio_soft, ratio_err_low_soft, ratio_err_high_soft = compute_ratios(color_flux_data["%s" % soft_band],
                                                                             color_flux_data["%s_lower" % soft_band],
                                                                             color_flux_data["%s_upper" % soft_band], color_flux_data["%s" % mid_band],
                                                                             color_flux_data["%s_lower" % mid_band], color_flux_data["%s_upper" % mid_band])

        markers = get_markers_array(len(color_flux_data["epoch"]))
        colors = create_color_array(len(color_flux_data["epoch"]), "plasma")

        print("Found %d observations" % len(color_flux_data))

        for index, color in enumerate(colors):
            if color_flux_data["xmm_obsid"][index] != "":
                label = color_flux_data["epoch"][index]
                if color_flux_data["nustar_obsid"][index] != "":
                    facemarkercolor = "None"
                else:
                    facemarkercolor = color
            else:
                label = color_flux_data["epoch"][index]
                facemarkercolor = "black"
            flux_ax.errorbar(ratio_soft[index], ratio_hard[index], xerr=[[ratio_err_low_soft[index]], [ratio_err_high_soft[index]]],
                             yerr=[[ratio_err_low_hard[index]], [ratio_err_high_hard[index]]],
                             label=label, markerfacecolor=facemarkercolor, color=color, marker=markers[index], fmt="-")
        flux_ax.set_xlabel("Softness")
        flux_ax.set_ylabel("Hardness")
        plt.tight_layout()
        flux_ax.legend()
        flux_figure.savefig("%s/%ssoftness_hardness.png" % (source_dir, plot_source), bbox_inches='tight', format="png")


    # flux component over times
    components = ["simpl_0", "diskbb_0", "diskbb_1", "diskpbb_0"]
    colordiskbb0 = "blue"
    colordiskbb1 = "red"
    colordiskpbb0 = "black"
    simplcolor = "green"
    colors = [simplcolor, colordiskbb0, colordiskbb1, colordiskpbb0]
    time_figure, flux_comp_ax = plt.subplots(1, 1)
    only_one_comp = 0
    param_2 = "flux"
    suffix = "0011000"
    for comp, color in zip(components, colors):
        print("Plotting %s luminosity vs epoch" % comp)
        comp_flux_file = "%s/fluxes/flux_%s.dat" % (source_dir, comp)

        if not os.path.isfile("%s" % comp_flux_file):
            print("%s not found for this source" % (comp_flux_file))
            continue

        flux_data = read_flux_comp("%s" % comp_flux_file)

        flux_data.sort(order="epoch")
        # store simpl observations
        if "simpl" in comp:
            simpl_obs_xmm = flux_data["xmm_obsid"]
            simpl_obs_chandra = flux_data["chandra"]

        # transform fluxes to L
        flux_data["%s%s" % (param_2, suffix)] = flux_data["%s%s" % (param_2, suffix)] * constant
        flux_data["%supp%s" % (param_2, suffix)] = flux_data["%supp%s" % (param_2, suffix)] * constant
        flux_data["%slow%s" % (param_2, suffix)] = flux_data["%slow%s" % (param_2, suffix)] * constant

        param_2err_low, param_2err_high, param_2lolimits, param_2uplimits = bounds_to_errors(flux_data["%s%s" % (param_2, suffix)],
                                                                                             flux_data["%slow%s" % (param_2, suffix)],
                                                                                             flux_data["%supp%s" % (param_2, suffix)])

        print("Found %d observations" % len(flux_data))

        for index, epoch in enumerate(flux_data["epoch"]):
            if only_one_comp:

                if flux_data["xmm_obsid"][index] != "" and flux_data["xmm_obsid"][index] in simpl_obs_xmm and "diskbb_1" == comp:
                    continue
                elif flux_data["chandra"][index] != "" and flux_data["chandra"][index] in simpl_obs_chandra and "diskbb_1" == comp:
                    print("continue %s chandra" % epoch)
                    continue
            if flux_data["xmm_obsid"][index] != "":
                label = flux_data["epoch"][index]
                if flux_data["nustar_obsid"][index] != "":
                    facemarkercolor = "white"
                    marker = "^"
                else:
                    facemarkercolor = color
                    marker = "o"
            else:
                facemarkercolor = "None"
                marker = "s"
            flux_comp_ax.errorbar(Time(epoch).jd, flux_data["%s%s" % (param_2, suffix)][index],
                                  yerr=[[param_2err_low[index]], [param_2err_high[index]]],
                                  marker=marker, fmt="-", ls="None",
                                  uplims=param_2uplimits[index], lolims=param_2lolimits[index],
                                  markerfacecolor=facemarkercolor, color=color)

    legend_elements = [Line2D([0], [0], marker='o', color=colordiskbb0, label='Soft diskbb', markersize=15),
                       Line2D([0], [0], marker='o', color=colordiskbb1, label='Hard diskbb', markersize=15),
                       Line2D([0], [0], marker='o', color=simplcolor, label='Simpl', markersize=15)]
    plt.legend(handles=legend_elements, title=source_name, title_fontsize=24, loc='best', frameon=False, fancybox=False, shadow=False)
    date_formatter = FuncFormatter(jd_to_daymonthyear)
    plt.locator_params(axis='x', nbins=6)
    plt.xticks(rotation=0)
    # save components flux plot
    flux_comp_ax.xaxis.set_major_formatter(date_formatter)
    #plt.show()
    plt.xlabel("Time")
    plt.ylabel("L$_{unabs}$ (10$^{39}$ erg/s)")
    time_figure.savefig("%s/comp_flux_time.png" % source_dir, format="png", bbox_inches='tight')
    #plt.show()

    # plot disks correlations
    components = ["diskbb_0", "diskbb_1", "diskpbb_0", "simpl_0", "bbody_0", "powerlaw_0", "diskbb_0", "diskbb_0"]
    ylabels = ["soft", "hard", "", "", "", "", "soft"]
    xlabels = ["T$_{in}$ (keV)", "T$_{in}$ (keV)", "T$_{in}$ (keV)", "$f_{scatt}$", "kT", "$\Gamma$", "norm", "radius"]
    params = ["Tin", "Tin", "Tin", "FracSctr", "kT", "PhoIndex", "norm", "radius"]
    for comp, ylabel, xlabel, param_1 in zip(components, ylabels, xlabels, params):
        print("Plotting %s vs luminosity" % comp)
        diskbb_0_file = "%s/fluxes/flux_%s.dat" % (source_dir, comp)
        diskbb_0_params_file = "%s/components/%s.dat" % (source_dir, comp)
        chandra_diskbb_0_params_file = "%s/components/chandra_%s.dat" % (source_dir, comp)

        param_2 = "flux"
        suffix = "0011000"

        diskbb_0_figure, diskbb_0_ax = plt.subplots(1, 1)
        diskbb_0_ax.set_xscale("log")
        diskbb_0_ax.set_yscale("log")
        formatter_major = FuncFormatter(lambda y, _: '%d' % y)
        formatter_minor = FuncFormatter(lambda y, _: '%.1f' % y)
        diskbb_0_ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        diskbb_0_ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        diskbb_0_ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        if not os.path.isfile("%s" % diskbb_0_file) or not os.path.isfile("%s" % diskbb_0_params_file):
            print("%s or %s not found for this source" % (diskbb_0_file, diskbb_0_params_file))
            continue

        flux_diskbb_data = read_flux_comp("%s" % diskbb_0_file)
        params_diskbb_data = read_params_diskbb("%s" % diskbb_0_params_file)
        if os.path.isfile("%s" % chandra_diskbb_0_params_file):
            chandra_params_diskbb_data = read_params_diskbb("%s" % chandra_diskbb_0_params_file)
            params_diskbb_data = np.append(params_diskbb_data, chandra_params_diskbb_data)

        params_diskbb_data.sort(order="epoch")

        # transform fluxes to L
        flux_diskbb_data["%s%s" % (param_2, suffix)] = flux_diskbb_data["%s%s" % (param_2, suffix)] * constant
        flux_diskbb_data["%supp%s" % (param_2, suffix)] = flux_diskbb_data["%supp%s" % (param_2, suffix)] * constant
        flux_diskbb_data["%slow%s" % (param_2, suffix)] = flux_diskbb_data["%slow%s" % (param_2, suffix)] * constant

        param_1err_low, param_1err_high, param_1lolimits, param_1uplimits = bounds_to_errors(params_diskbb_data["%s" % param_1],
                                                                                             params_diskbb_data["%slow" % param_1],
                                                                                             params_diskbb_data["%supp" % param_1])
        param_2err_low, param_2err_high, param_2lolimits, param_2uplimits = bounds_to_errors(flux_diskbb_data["%s%s" % (param_2, suffix)],
                                                                                             flux_diskbb_data["%slow%s" % (param_2, suffix)],
                                                                                             flux_diskbb_data["%supp%s" % (param_2, suffix)])

        markers = get_markers_array(len(flux_diskbb_data["epoch"]))
        colors = create_color_array(len(flux_diskbb_data["epoch"]), "plasma")
        diskbb_0_ax.set_prop_cycle(cycler('color', colors))
        print("Found %d observations" % len(flux_diskbb_data))

        for index, color in enumerate(colors):
            if flux_diskbb_data["xmm_obsid"][index] != "":
                label = flux_diskbb_data["epoch"][index]
                if flux_diskbb_data["nustar_obsid"][index] != "":
                    facemarkercolor = "None"
                else:
                    facemarkercolor = color
            else:
                label = flux_diskbb_data["epoch"][index]
                facemarkercolor = "black"

            diskbb_0_ax.errorbar(params_diskbb_data["%s" % param_1][index], flux_diskbb_data["%s%s" % (param_2, suffix)][index],
                                 xerr=[[param_1err_low[index]], [param_1err_high[index]]],
                                 yerr=[[param_2err_low[index]], [param_2err_high[index]]], label=flux_diskbb_data["epoch"][index],
                                 marker=markers[index], fmt="-", ls="None", xlolims=param_1lolimits[index], xuplims=param_1uplimits[index],
                                 uplims=param_2uplimits[index], lolims=param_2lolimits[index],
                                 markerfacecolor=facemarkercolor)

        diskbb_0_ax.set_xlabel("%s" % xlabel)
        diskbb_0_ax.set_ylabel("L$_{bol}$ %s diskbb (10$^{39}$ erg/s)" % ylabel)
        plt.tight_layout()
        plt.legend(fontsize=12)
        #plt.legend()
        plt.title(source_name)
        fig_name = "%s/%s%s%s_L.png" % (source_dir, plot_source, comp, param_1)
        diskbb_0_figure.savefig(fig_name, bbox_inches='tight', format="png")
        print("Saved figure %s" % fig_name)
        if "simpl" in comp or "powerlaw" in comp:
            continue

        # remove upper limits or lower limits
        temperature_nolimits = np.array([data for data in params_diskbb_data if not data["%slow" % param_1] == 0 and not data["%supp" % param_1] == 0])
        
        flux_nolimits = np.array([data for data in flux_diskbb_data if not data["%supp%s" % (param_2, suffix)] == 0 and not data["%supp%s" % (param_2, suffix)] == 0])
        if len(temperature_nolimits) > len(flux_nolimits):
            temperature_nolimits = np.array([temp for temp in temperature_nolimits if temp["xmm_obsid"] in flux_nolimits["xmm_obsid"] and temp["chandra"] in flux_nolimits["chandra"]])
        else:
            flux_nolimits = np.array([flux for flux in flux_nolimits if flux["xmm_obsid"] in temperature_nolimits["xmm_obsid"] and flux["chandra"] in temperature_nolimits["chandra"]])

        # there are no limits now
        param_1err_low, param_1err_high, _, _ = bounds_to_errors(temperature_nolimits["%s" % param_1],
                                                                 temperature_nolimits["%slow" % param_1],
                                                                 temperature_nolimits["%supp" % param_1])

        param_2err_low, param_2err_high, _, _ = bounds_to_errors(flux_nolimits["%s%s" % (param_2, suffix)],
                                                                 flux_nolimits["%slow%s" % (param_2, suffix)],
                                                                 flux_nolimits["%supp%s" % (param_2, suffix)])

        rho, p = spearmanr(temperature_nolimits["%s" % param_1], flux_nolimits["%s%s" % (param_2, suffix)])
        try:
            rho, p = spearmanr(temperature_nolimits["%s" % param_1], flux_nolimits["%s%s" % (param_2, suffix)])
        except ValueError:
            print("Error: number of datapoints is not same for temperature and luminosity. Skipping")
            continue

        a, b, a_err, b_err = fit_diskbb_odr(temperature_nolimits["%s" % param_1], param_1err_low, param_1err_high, flux_nolimits["%s%s" % (param_2, suffix)],
                                            param_2err_low, param_2err_high, index=2 * rho, norm=5)
        f = open("%s/corr_%s%s.txt" % (source_dir, comp, param_1), "w+")
        f.write("#rho\tp\ta\ta_err\tb\tb_err\n&%.2f\t&%.3f\t&%.2f\t&%.2f\t&%.2f\t&%.2f" % (rho, p, a, a_err, b, b_err))
        f.close()
        print("Rho %.2f" % rho)
        print("P value (although not to be trusted) %.5f" % p)
        print("Correlation index %.2f $\pm$ %.2f" % (b, b_err))

        xlims = diskbb_0_ax.get_xlim()
        plt.plot(diskbb_0_ax.get_xlim(), diskLvsT(diskbb_0_ax.get_xlim(), a, b),
                 color="black", marker=None, ls='solid', label='L $\propto$ T$^{%.1f \pm %.1f}$' % (b, b_err))
        plt.legend()
        diskbb_0_figure.canvas.draw()
        diskbb_0_ax.set_xlim(xlims)

        #plt.fill_between(plt.xlim(), diskLvsT(plt.xlim(), a, b + b_err),
        #                 diskLvsT(plt.xlim(), a, b - b_err), color="black", alpha="0.2")

        #diskbb_0_ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        #diskbb_0_ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        #diskbb_0_ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())

        diskbb_0_figure.savefig("%s/%s%s%s_L_corr.png" % (source_dir, plot_source, comp, param_1), bbox_inches='tight', format="png")
    # total flux vs R component

    total_band = "0330"
    if not os.path.isfile("%s" % flux_file):
        print("Fluxes not found for this source")
        continue

    flux_data = read_newflux_data("%s" % flux_file)
    components = ["diskbb_0", "diskbb_1"]
    xlabels = ["Soft norm ", "Hard norm"]

    for comp, xlabel in zip(components, xlabels):
        flux_size_figure, size_ax = plt.subplots(1, 1)
        print("Plotting %s vs luminosity" % comp)
        diskbb_0_file = "%s/fluxes/flux_%s.dat" % (source_dir, comp)
        diskbb_0_params_file = "%s/components/%s.dat" % (source_dir, comp)
        chandra_diskbb_0_params_file = "%s/components/chandra_%s.dat" % (source_dir, comp)

        param_1 = "norm"

        diskbb_0_figure, diskbb_0_ax = plt.subplots(1, 1)
        if not os.path.isfile("%s" % diskbb_0_file) or not os.path.isfile("%s" % diskbb_0_params_file):
            print("%s or %s not found for this source" % (diskbb_0_file, diskbb_0_params_file))
            continue

        params_diskbb_data = read_params_diskbb("%s" % diskbb_0_params_file)

        if os.path.isfile("%s" % chandra_diskbb_0_params_file):
            chandra_params_diskbb_data = read_params_diskbb("%s" % chandra_diskbb_0_params_file)
            params_diskbb_data = np.append(params_diskbb_data, chandra_params_diskbb_data)

        param_1err_low, param_1err_high, param_1lolimits, param_1uplimits = bounds_to_errors(params_diskbb_data["%s" % param_1],
                                                                                             params_diskbb_data["%slow" % param_1],
                                                                                             params_diskbb_data["%supp" % param_1])

        colors = create_color_array(len(params_diskbb_data), "plasma")
        size_ax.set_prop_cycle(cycler('color', colors))
        print("Found %d observations" % len(params_diskbb_data))

        for index, color in enumerate(colors):
            if params_diskbb_data["xmm_obsid"][index] != "":
                label = params_diskbb_data["epoch"][index]
                if params_diskbb_data["nustar_obsid"][index] != "":
                    facemarkercolor = "None"
                else:
                    facemarkercolor = color
            else:
                label = flux_diskbb_data["epoch"][index]
                facemarkercolor = "black"
            size_ax.errorbar(params_diskbb_data["%s" % param_1][index], constant * flux_data["%s" % total_band][index],
                             xerr=[[param_1err_low[index]], [param_1err_high[index]]],
                             yerr=[[constant * (flux_data["%s" % total_band] - flux_data["%s_lower" % total_band])[index]],
                             [constant * (flux_data["%s_upper" % total_band] - flux_data["%s" % total_band])[index]]],
                             label=label, markerfacecolor=facemarkercolor, color=color,
                             marker=markers[index], fmt="-", xlolims=param_1lolimits[index], xuplims=param_1uplimits[index])
        a, b, a_err, b_err = fit_diskbb_odr(params_diskbb_data["%s" % param_1], param_1err_low, param_1err_high,
                                            constant * flux_data["%s" % total_band], constant * (flux_data["%s" % total_band] - flux_data["%s_lower" % total_band]),
                                            constant * (flux_data["%s_upper" % total_band] - flux_data["%s" % total_band]), index=-2/7, norm=200)
        print("Index: %.2f (%.2f)" % (b, b_err))
        size_ax.plot(size_ax.get_xlim(), diskLvsT(size_ax.get_xlim(), a, b),
                     color="black", marker=None, ls='solid', label='L $\propto$ T$^{%.1f \pm %.1f}$ ($\\rho$ )' % (b, b_err))

        size_ax.set_xlabel(xlabel)
        size_ax.set_ylabel("L [0.3 - 30 keV] (10$^{39}$)")
        size_ax.legend()
        flux_size_figure.savefig("%s/%sflux_size.png" % (source_dir, comp), bbox_inches='tight', format="png")
