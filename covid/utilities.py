import csv
import json
import numpy as np
from scipy import interpolate, stats
import math
from scipy.optimize import curve_fit
from datetime import datetime
from decimal import Decimal
from sys import stdout
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(stdout)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_logger():
    return logger


def date_to_int(d):
    return int(datetime.strftime(d, '%Y%m%d'))


def int_to_date(d):
    return datetime.strptime(str(d), '%Y%m%d').date()


def doubling_time(case_data):
    days = len(case_data)
    Td = days / np.log2(case_data[-1] / case_data[0])
    return Td


def json_handler(o):
    if isinstance(o, datetime):
        return o.isoformat()
    elif isinstance(o, Decimal):
        return float(o)
    else:
        logger.warning('Unknown Type in json_handler: ' + str(o))
        return str(o)


def csv_to_json(infile, outfile):
    csvFile = open(infile, 'r')
    j = dict()
    csvReader = csv.DictReader(csvFile)
    for l in csvReader:
        state = l['State']
        county = l['County']
        pop = int(l['Population'])
        if state not in j:
            j[state] = {}
            j[state][county] = pop
        else:
            j[state][county] = pop
    jsonFile = open(outfile, 'w')
    jsonFile.write(json.dumps(j))


def interp_mat(M, old_domain, new_domain, rows, cols):
    lons = np.linspace(old_domain[0], old_domain[1], M.shape[1])
    lats = np.linspace(old_domain[3], old_domain[2], M.shape[0])
    f = interpolate.interp2d(lons, lats, M, kind = 'cubic')
    lons_new = np.linspace(new_domain[0], new_domain[1], cols)
    lats_new = np.linspace(new_domain[3], new_domain[2], rows)
    return f(lons_new, lats_new)


def laplacian(M, dx, dy):
    rows, cols = M.shape
    L = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):

            #interior
            if 1 <= i <= (rows-2) and 1 <= j <= (cols-2):
                L[i, j] = (M[i-1, j]+M[i+1, j]-2*M[i, j])/(dy*dy)+(M[i, j-1]+M[i, j+1]-2*M[i, j])/(dx*dx)

            # sides
            if i == 0 and 1 <= j <= (cols-2):
                L[i, j] = (M[i+1, j]-2*M[i, j])/(dy*dy)+(M[i, j-1]+M[i, j+1]-2*M[i, j])/(dx*dx)
            if i == rows-1 and 1 <= j <= (cols-2):
                L[i, j] = (M[i-1, j]-2*M[i, j])/(dy*dy)+(M[i, j-1]+M[i, j+1]-2*M[i, j])/(dx*dx)
            if j == 0 and 1 <= i <= (rows-2):
                L[i, j] = (M[i-1, j]+M[i+1, j]-2*M[i, j])/(dy*dy)+(M[i, j+1]-2*M[i, j])/(dx*dx)
            if j == cols-1 and 1 <= i <= (rows-2):
                L[i, j] = (M[i-1, j]+M[i+1, j]-2*M[i, j])/(dy*dy)+(M[i, j-1]-2*M[i, j])/(dx*dx)

            # corners
            if i == 0 and j == 0:
                L[i, j] = (M[i+1, j]-2*M[i, j])/(dy*dy)+(M[i, j+1]-2*M[i, j])/(dx*dx)
            if i == 0 and j == cols-1:
                L[i, j] = (M[i+1, j]-2*M[i, j])/(dy*dy)+(M[i, j-1]-2*M[i, j])/(dx*dx)
            if i == rows-1 and j == 0:
                L[i, j] = (M[i-1, j]-2*M[i, j])/(dy*dy)+(M[i, j+1]-2*M[i, j])/(dx*dx)
            if i == rows-1 and j == cols-1:
                L[i, j] = (M[i-1, j]-2*M[i, j])/(dy*dy)+(M[i, j-1]-2*M[i, j])/(dx*dx)
    return L


def vec_diff(v):
    tmp = [v[0]-(v[1]-v[0])]+list(v)
    diff = [tmp[i+1]-tmp[i] for i in range(len(v))]
    return diff


def get_beta_ramp(p):
    beta_array = []
    for i in range(p.n_days):
        if i<p.Sd_delay:
            tmp = p.beta
        elif p.Sd_delay <= i<p.Sd_delay+p.n_ramp:
            tmp = p.beta*(1-p.Sd*(float((i-p.Sd_delay)/p.n_ramp)))
        else:
            tmp = p.beta*(1-p.Sd)
        beta_array.append(tmp)

    return beta_array


def get_seasonal_beta_variation(date, peak_date, amp):
    var = amp*math.cos(2*math.pi(date-peak_date).days/365.0)
    return var


def cost_func(v0, v1):
    avg_step = 3
    tmp0 = [np.mean(v0[i: i+avg_step]) for i in range(len(v0)-avg_step+1)]
    tmp1 = v1[: len(tmp0)]
    min_len = min((len(tmp0), len(tmp1)))
    return np.sqrt(np.sum([((tmp1[i]-tmp0[i]))**2 for i in range(min_len)]))/(min_len)


def grad_descent(p0, J0, J1, dp, alpha):
    new_p = {}
    for k in p0:
        new_p[k] = p0[k]-alpha[k]*(J1[k]-J0[k])/dp[k]
    return new_p


def exp_func(t, a, b):
    return a*np.exp(b*t)


def beta_interp(t, beta, Sd, t0, n_ramp):
    if t<t0:
        return beta
    elif t0 <= t <= t0+n_ramp:
        return (1-(t-t0)*Sd/(n_ramp))*beta
    else:
        return (1-Sd)*beta


def exp_decay_func(t, a, beta, Sd, t0, n_ramp):

    return [a*np.exp(i*beta_interp(i, beta, Sd, t0, n_ramp)) for i in t]


def exp_lin_func(t, a, b, c):
    return a*np.exp(b*t)+c*t


def logistic_func(t, a, b, c, d, e):
    return a/(d*np.exp(-b*(t-e))+c)


class initial_values:
    pass


def cases_to_beta(data, params):

    popt, pconv = curve_fit(exp_func, [i for i in range(len(data))], data)

    params.R0 = (popt[1]/params.gamma+1)
    params.beta = params.R0*params.gamma

    return params


def doubling_trend_to_Sd(Td_array):
    days = [i for i in range(len(Td_array))]
    r = stats.linregress(days, Td_array)
    Sd = r.slope/(1+r.slope)
    return Sd
