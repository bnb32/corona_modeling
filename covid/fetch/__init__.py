"""Fetch data"""
from datetime import date, timedelta
from os import environ, path
import urllib.request
from tempfile import gettempdir
import json
import numpy as np
import pandas as pd

import covid.environment as env
from covid.utilities import get_logger, int_to_date, date_to_int


logger = get_logger()

pop_path = path.join(environ['LAMBDA_TASK_ROOT'], 'pop.json')
states_path = path.join(environ['LAMBDA_TASK_ROOT'], 'us_states.json')

cases_path = path.join(gettempdir(),
                       'case_data_%s-%s-%s.json' % (date.today().day,
                                                    date.today().month,
                                                    date.today().year))

params_path = path.join(gettempdir(),
                        'params_%s-%s-%s.json' % (date.today().day,
                                                  date.today().month,
                                                  date.today().year))

orders_path = path.join(environ['LAMBDA_TASK_ROOT'], 'sah_order_dates.json')

api_key = env.GOOGLE_API_KEY

"""Use estimated death rate and hospitalization rate
   to estimate infections and recovered
"""

estimation_params = {}
estimation_params['death_rate'] = 0.02
estimation_params['hospitalization_rate'] = 0.005


def artificial_V(n_days, N):

    V_final = N*estimation_params['vaccination_rate']
    V_initial = 0
    return np.linspace(V_initial, V_final, n_days)


class Dataset:
    """Dataset class for getting and
       formatting covid data for processing
       and simluation by model
    """
    def __init__(self, parameters=None):
        self.current_url = 'https: //api.covidactnow.org/v2/states.json?'
        self.current_url += f'apiKey = {env.COVID_ACT_API}'
        self.timeseries_url = 'https: //api.covidactnow.org/v2/'
        self.timeseries_url += 'states.timeseries.json?'
        self.timeseries_url += f'apiKey = {env.COVID_ACT_API}'
        self.data = None

        with open(pop_path, 'rt') as pop_json:
            self.pop_data = json.load(pop_json)
        with open(states_path, 'rt') as states_json:
            self.us_states = json.load(states_json)
        with open(orders_path, 'rt') as orders_json:
            self.order_dates = json.load(orders_json)

        self.params = {}
        self.define_parameters()

        if parameters is not None:
            for k, v in parameters.items():
                self.params[k] = v

        self.params['N'] = self.get_pop()

    def define_parameters(self):
        self.params['state'] = 'Washington'
        self.params['county'] = 'Washington'
        self.params['n_days'] = 100

    def compartment_map(self):
        self.data = {}
        self.data['H'] = self.raw_data['hospital'].values+self.raw_data['icu'].values
        self.data['D'] = self.raw_data['deaths'].values
        self.data['R'] = self.data['D']/estimation_params['death_rate']
        self.data['I'] = self.raw_data['cases'].values-self.data['R']-self.data['D']
        self.data['V'] = self.raw_data['vaccinationsCompleted'].values
        self.data['S'] = self.params['N']-self.data['I']-self.data['R']-self.data['V']-self.data['D']

        return self.data

    def get_data(self):
        self.get_raw_data()
        return self.compartment_map()

    def get_raw_data(self):
        data = pd.read_json(self.timeseries_url)
        data_metrics = pd.DataFrame(data[data['state'] == self.us_states[self.params['state']]]['metricsTimeseries'].values[0])
        data = pd.DataFrame(data[data['state'] == self.us_states[self.params['state']]]['actualsTimeseries'].values[0])
        data['infections'] = data_metrics['caseDensity'].apply(lambda x: x*self.params['N']/10**5)
        data['date'] = data['date'].apply(lambda x: int(x.replace('-', '')))
        self.params['min_date'] = int_to_date(max(data['date'].values))-timedelta(days = self.params['n_days'])
        self.params['max_date'] = int_to_date(max(data['date'].values))
        data = data[(data['date'] > date_to_int(self.params['min_date'])) &
                    (data['date'] < date_to_int(self.params['max_date']))]
        data['hospital'] = data['hospitalBeds'].apply(lambda x: x['currentUsageTotal'])
        data['icu'] = data['icuBeds'].apply(lambda x: x['currentUsageTotal'])
        self.raw_data = data.sort_values(by = 'date', ascending = True)
        self.raw_data = self.raw_data.fillna(self.raw_data.rolling(6, min_periods = 1).mean())
        self.raw_data = self.raw_data.fillna(0)
        self.raw_data = self.raw_data.ewm(span = 10).mean()
        return self.raw_data

    def get_pop(self):
        data = pd.read_json(self.current_url)
        return data[data['state'] == self.us_states[self.params['state']]]['population'].values[0]
        #if self.params['county'] == None:
        #    self.county = state
        #return self.pop_data[self.params['state']][self.params['county']]

def latlon_to_place(api_key, lat, lon):
    uri = 'https: //maps.googleapis.com/maps/api/geocode/json?latlng = {Lat}, {Lon}&key = {ApiKey}'.format(
        ApiKey = api_key,
        Lat = lat,
        Lon = lon,
    )

    logger.info('Retrieving GeoCoding for ({0}, {1})...'.format(lat, lon))

    v = urllib.request.urlopen(uri).read()
    logger.info(v)
    j = json.loads(v)

    components = j['results'][0]['address_components']
    country = state = county = None
    for c in components:
        if "country" in c['types']:
            country = c['long_name']
        if "administrative_area_level_1" in c['types']:
            state = c['long_name']
        if "administrative_area_level_2" in c['types']:
            county = c['long_name']

    return county, state, country

