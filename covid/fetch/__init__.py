import covid.environment as env
from covid.misc import get_logger, int_to_date, date_to_int

from datetime import date, timedelta, datetime
from os import environ, path, system
import json
import numpy as np
import urllib.request
from scipy import stats
from tempfile import gettempdir
import pandas as pd

logger = get_logger()

pop_path = path.join(environ['LAMBDA_TASK_ROOT'], 'pop.json')
states_path = path.join(environ['LAMBDA_TASK_ROOT'], 'us_states.json')

cases_path = path.join(gettempdir(), 'case_data_%s-%s-%s.json'%(date.today().day,date.today().month,date.today().year))

params_path = path.join(gettempdir(), 'params_%s-%s-%s.json'%(date.today().day,date.today().month,date.today().year))

orders_path = path.join(environ['LAMBDA_TASK_ROOT'], 'sah_order_dates.json')

api_key=env.google_api_key
covid_uri=env.covid_uri

"""Use estimated death rate and hospitalization rate
   to estimate infections and recovered
"""

estimation_params = {}
estimation_params['death_rate'] = 0.01
estimation_params['hospitalization_rate'] = 1.0/6.0
estimation_params['vaccination_rate'] = 0.6

def artificial_V(n_days,N):

    V_final = N*estimation_params['vaccination_rate']
    V_initial = 0
    return np.linspace(V_initial,V_final,n_days)

class Dataset:
    """Dataset class for getting and
       formatting covid data for processing
       and simluation by model
    """   
    def __init__(self,parameters=None):
        self.uri = env.covid_uri
        self.url = "https://covidtracking.com/api/v1/states/daily.json"

        with open(pop_path, 'rt') as pop_json:
            self.pop_data = json.load(pop_json)
        with open(states_path, 'rt') as states_json:
            self.us_states = json.load(states_json)
        with open(orders_path, 'rt') as orders_json:
            self.order_dates = json.load(orders_json)
        
        self.params = {}
        self.define_parameters()

        if parameters is not None:
            for k,v in parameters.items():
                self.params[k] = v
        
        self.params['N'] = self.get_pop()
    
    def define_parameters(self):
        self.params['state'] = 'Washington'
        self.params['county'] = 'Washington'
        self.params['n_days'] = 100
    
    def compartment_map(self):    
        self.data = {}
        self.data['H'] = self.raw_data['hospitalizedCurrently'].values                 
        self.data['I'] = self.data['H']+self.data['H']/estimation_params['hospitalization_rate']
        self.data['D'] = self.raw_data['death'].values
        self.data['R'] = self.data['D']+self.data['D']/estimation_params['death_rate']
        self.data['V'] = artificial_V(self.params['n_days'],self.params['N'])
        self.data['S'] = self.params['N']-self.data['I']-self.data['R']-self.data['V'] 

        return self.data
    
    def get_data(self):
        self.get_raw_data()
        return self.compartment_map()
        
    def get_raw_data(self):
        data = pd.read_json(self.url)
        data = data[(data['state']==self.us_states[self.params['state']])]
        self.params['min_date'] = int_to_date(max(data['date'].values))-timedelta(days=self.params['n_days'])
        data = data[(data['date'] > date_to_int(self.params['min_date']))]
        self.raw_data = data.sort_values(by='date',ascending=True)
        self.raw_data.fillna(0,inplace=True)
        return self.raw_data

    def get_pop(self):
        if self.params['county']==None: 
            self.county=state
        return self.pop_data[self.params['state']][self.params['county']] 
 
def latlon_to_place(api_key, lat, lon):
    uri = 'https://maps.googleapis.com/maps/api/geocode/json?latlng={Lat},{Lon}&key={ApiKey}'.format(
        ApiKey=api_key,
        Lat=lat,
        Lon=lon,
    )
    
    logger.info('Retrieving GeoCoding for ({0}, {1})...'.format(lat, lon))
    
    v=urllib.request.urlopen(uri).read()
    logger.info(v)
    j=json.loads(v)
    
    components=j['results'][0]['address_components']
    country=state=county=None
    for c in components:
        if "country" in c['types']:
            country=c['long_name']
        if "administrative_area_level_1" in c['types']:
            state=c['long_name']
        if "administrative_area_level_2" in c['types']:
            county=c['long_name']

    return county,state,country        

