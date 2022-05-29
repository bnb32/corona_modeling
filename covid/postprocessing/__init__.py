import covid.environment as env
from covid.utilities import get_logger

import os
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
import urllib.parse, urllib.request
from math import log, exp, tan, atan, pi
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from os import path

logger = get_logger()

long_names = {'S': 'Susceptible',
              'E': 'Exposed',
              'A': 'Asymptomatic',
              'I': 'Infected',
              'H': 'Hospitalized',
              'Q': 'Quarantined',
              'R': 'Recovered',
              'D': 'Deaths',
              'V': 'Vaccinated'}

def plot_compartment(sim_data = None, raw_data = None, params = None, compartment = None):

    fig = plt.figure()

    plt.title(f'{params["state"]}')

    data = np.concatenate((raw_data[compartment], sim_data[compartment]))
    dates = get_date_labels(n_days = len(data), min_date = params['min_date'])
    x_values = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]

    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval = len(dates)//10)
    ax.xaxis.set_major_locator(locator)

    plt.plot(x_values, data, label = f'{long_names[compartment]}')

    plt.axvline(x = x_values[params['n_days']-1], color = 'black')
    plt.xticks(rotation = 60)
    plt.legend()
    plt.tight_layout()

    filename = path.join(env.COVID_FIG_DIR, f'{params["state"]}_{compartment}_fig.png')
    plt.savefig(filename)
    print(f'Saved figure: {filename}')

def plot_compartment_comparison(sim_data = None, raw_data = None, params = None, compartment = None):

    fig = plt.figure()

    plt.title(f'{params["state"]}')

    n_days = max((len(raw_data[compartment]), len(sim_data[compartment])))
    dates = get_date_labels(n_days = n_days, min_date = params['min_date'])
    x_values = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]

    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval = len(dates)//10)
    ax.xaxis.set_major_locator(locator)

    plt.plot(x_values[: len(sim_data[compartment])], sim_data[compartment], label = f'{long_names[compartment]} - Simulation')
    plt.plot(x_values[: len(raw_data[compartment])], raw_data[compartment], label = f'{long_names[compartment]} - Data')

    plt.xticks(rotation = 60)
    plt.legend()
    plt.tight_layout()

    filename = path.join(env.COVID_FIG_DIR, f'{params["state"]}_{compartment}_comparison_fig.png')
    plt.savefig(filename)
    print(f'Saved figure: {filename}')

def plot_compartments(sim_data = None, raw_data = None, params = None):

    fig = plt.figure()

    plt.title(f'{params["state"]}')

    first_key = list(sim_data.keys())[0]
    data = np.concatenate((raw_data[first_key], sim_data[first_key]))
    dates = get_date_labels(n_days = len(data), min_date = params['min_date'])
    x_values = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]

    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval = len(dates)//10)
    ax.xaxis.set_major_locator(locator)

    for i, k in enumerate(sim_data):
        data = np.concatenate((raw_data[k], sim_data[k]))
        plt.plot(x_values, data, label = f'{long_names[k]}')

    plt.xticks(rotation = 60)
    plt.axvline(x = x_values[params['n_days']-1], color = 'black')
    plt.legend()
    plt.tight_layout()

    filename = path.join(env.COVID_FIG_DIR, f'{params["state"]}_fig.png')
    plt.savefig(filename)
    print(f'Saved figure: {filename}')

def plot_compartments_comparison(sim_data = None, raw_data = None, params = None):

    fig = plt.figure()

    plt.title(f'{params["state"]}')

    first_key = list(sim_data.keys())[0]
    n_days = max((len(raw_data[first_key]), len(sim_data[first_key])))
    dates = get_date_labels(n_days = n_days, min_date = params['min_date'])
    x_values = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]

    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval = len(dates)//10)
    ax.xaxis.set_major_locator(locator)

    for i, k in enumerate(sim_data):
        plt.plot(x_values[: len(sim_data[k])], sim_data[k], label = f'{long_names[k]} - Simulation')
        plt.plot(x_values[: len(raw_data[k])], raw_data[k], label = f'{long_names[k]} - Data')

    plt.xticks(rotation = 60)
    plt.legend()
    plt.tight_layout()

    filename = path.join(env.COVID_FIG_DIR, f'{params["state"]}_comparison_fig.png')
    plt.savefig(filename)
    print(f'Saved figure: {filename}')

def get_date_labels(n_days = None, min_date = None):

    dates = []
    for i in range(n_days):
        day = (min_date+timedelta(days = i)).strftime('%Y-%m-%d')
        dates+= [day]
    return dates

def get_colormap():

    # get colormap
    cmap = plt.cm.jet
    color_array = cmap(np.arange(cmap.N))

    # change alpha values
    color_array[: , -1] = np.linspace(0.0, 1.0, cmap.N)

    # create a colormap object
    my_cmap = ListedColormap(color_array)

    return my_cmap

def get_map(domain):

    lon_min = domain[0]
    lon_max = domain[1]
    lat_min = domain[2]
    lat_max = domain[3]
    lat_mid = (lat_min+lat_max)/2
    lon_mid = (lon_min+lon_max)/2

    upperleft = '%s, %s'%(lat_max, lon_min)
    lowerright = '%s, %s'%(lat_min, lon_max)

    zoom = latlonrangetozoom(lat_min, lat_max, lon_min, lon_max)

    ullat, ullon = map(float, upperleft.split(', '))
    lrlat, lrlon = map(float, lowerright.split(', '))

    ulx, uly = latlontopixels(ullat, ullon, zoom)
    lrx, lry = latlontopixels(lrlat, lrlon, zoom)
    dx, dy = lrx - ulx, uly - lry
    position = '%s, %s'%(lat_mid, lon_mid)
    urlparams = urllib.parse.urlencode({'center': position,
                                  'zoom': str(zoom),
                                  'size': '%dx%d' %(dx, dy),
                                  'sensor': 'false',
                                  'key': api_key,
                                  'scale': 1})
    url = 'http: //maps.google.com/maps/api/staticmap?' + urlparams
    f = urllib.request.urlopen(url)
    im = Image.open(io.BytesIO(f.read()))

    return im

EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

def latlonrangetozoom(lat_min, lat_max, lon_min, lon_max):

    mx_min = (lon_min * ORIGIN_SHIFT) / 180.0
    my_min = log(tan((90 + lat_min) * pi/360.0))/(pi/180.0)
    my_min = (my_min * ORIGIN_SHIFT) /180.0

    mx_max = (lon_max * ORIGIN_SHIFT) / 180.0
    my_max = log(tan((90 + lat_max) * pi/360.0))/(pi/180.0)
    my_max = (my_max * ORIGIN_SHIFT) /180.0

    x_zoom = np.log2(640.0/(mx_max-mx_min)*INITIAL_RESOLUTION)
    y_zoom = np.log2(640.0/(my_max-my_min)*INITIAL_RESOLUTION)
    return np.min([int(x_zoom), int(y_zoom)])

def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

def pixelstolatlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon

def load_data(infile):
    f = open(infile, 'r')
    lines = f.readlines()
    data = []
    for l in lines:
        if "ncols" in l:
            ncols = int(l.strip('ncols').strip(' '))
        elif "nrows" in l:
            nrows = int(l.strip('nrows').strip(' '))
        elif "xllcorner" in l:
            xllcorner = int(l.strip('xllcorner').strip(' '))
        elif "yllcorner" in l:
            yllcorner = int(l.strip('yllcorner').strip(' '))
        elif "cellsize" in l:
            cellsize = float(l.strip('cellsize').strip(' '))
        elif "NODATA_value" in l:
            NODATA_value = int(l.strip('NODATA_value').strip(' '))
        else:
            data.append(list(l.split(' '))[0: -1])

    out = np.array(data, dtype = np.float)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i, j] == NODATA_value: out[i, j] = 0
    return out, xllcorner, yllcorner, cellsize, NODATA_value

def plot_doubling_trend(dates, data, state):
    days = [i for i in range(len(dates))]
    plt.plot(days, data)
    tick_step = max((1, int(len(days)/3)-1))
    plt.xticks(days[0: : tick_step], dates[0: : tick_step], rotation = 30)

    plt.title(state)
    plt.grid(True)
    plt.ylabel("Doubling Time (days)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(f'{env.COVID_FIG_DIR}/Td_trend.png')
    plt.clf()

