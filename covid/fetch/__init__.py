import covid.environment as env
from covid.misc import get_logger, params, int_to_date, date_to_int

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

class Dataset:
    def __init__(self,parameters=None):
        self.uri = env.covid_uri

        with open(pop_path, 'rt') as pop_json:
            self.pop_data = json.load(pop_json)
        with open(states_path, 'rt') as states_json:
            self.us_states = json.load(states_json)
        with open(orders_path, 'rt') as orders_json:
            self.order_dates = json.load(orders_json)
        
        if parameters is None:
            self.params = {}
            self.define_parameters()
        else:
            self.params = parameters
            self.params['N'] = self.get_pop()
    
    def compartment_map(self,data):    
        self.data_dict = {'S':[],
                          'E':[],
                          'A':[],
                          'Q':[],
                          'I':['positive'],
                          'H':['hospitalizedCurrently'],
                          'R':['recovered','hospitalizedDischarged'],
                          'D':['death']}
        return self.data_dict
        
    def define_parameters(self):

        self.params['state'] = 'Washington'
        self.params['county'] = 'Washington'
        self.params['n_days'] = 7
    
    def get_data(self):
        url="https://covidtracking.com/api/v1/states/daily.json"
        data = pd.read_json(url)
        data = data[(data['state']==self.us_states[self.params['state']])]
        min_date = int_to_date(max(data['date'].values))-timedelta(days=self.params['n_days'])
        self.data = data[(data['date'] > date_to_int(min_date))]
        return self.data

    def get_pop(self):
        if self.params['county']==None: 
            self.county=state
        return self.pop_data[self.params['state']][self.params['county']] 
 
def select_data(case_data):
    
    data = {}

    I_total=case_data["positive"].values
        
    I_new=case_data["positiveIncrease"].values

    D=case_data["death"].values
    
    R=case_data["recovered"].values 
    
    H_total=case_data["hospitalizedCumulative"].values
    
    H=case_data["hospitalizedCurrently"].values
     
    dates=[int_to_date(x) for x in case_data["date"].values]

    I_total[np.where(I_total==None)[0]]=0
    data['I_cum']=I_total[::-1]
    
    I_new[np.where(I_new==None)[0]]=0
    data['I_new']=I_new[::-1]
    
    D[np.where(D==None)[0]]=0
    data['D']=D[::-1]
    
    R[np.where(R==None)[0]]=0
    data['R']=R[::-1]
    
    H[np.where(H==None)[0]]=0
    data['H']=H[::-1]
    
    H_total[np.where(H_total==None)[0]]=0
    data['H_cum']=H_total[::-1]
    
    data['I_act']=data['I_cum']-data['R']-data['D']
    
    data['I']=data['I_act']

    data['dates']=dates[::-1]  
    return data

def refine_data(data,params):    

    #approx detected active cases
    tmp=(data['I_act']-data['H'])

    data['I_act']=data['I_act']/params['detection_rate']
    data['I_cum']=data['I_cum']/params['detection_rate']
    
    data['I_new']=data['I_new']/params['detection_rate']
    
    data['R']=data['R']/params['detection_rate']
    
    #approx undetected cases
    data['I_total']=tmp/params['detection_rate']
    
    data['A']=(data['I_total'])*params['A_to_I_ratio']/(1+params['A_to_I_ratio'])

    data['E']=np.array([0]*len(data['A']))
    data['I']=np.array([0]*len(data['A']))
    data['Q']=np.array([0]*len(data['A']))

    for n in range(1,len(data['A'])):
        data['I'][n]=data['I'][n-1]*(1-params['I_decay_rate'])+params['A_to_I_rate']*data['A'][n-1]
        if data['I'][n]<0: data['I'][n]=0
        data['Q'][n]=data['Q'][n-1]*(1-params['Q_decay_rate'])+params['I_to_Q_rate']*data['I'][n-1]
        if data['Q'][n]<0: data['Q'][n]=0
        
    return data

def data_to_initial_values(data,params):

    S = params.N-data['A'][-1]-data['I'][-1]-data['Q'][-1]
    S += -data['E'][-1]-data['R'][-1]-data['D'][-1]-data['H'][-1]
    #initialize compartments
    initial_values = {'E':data['E'][-1],
                      'Q':data['Q'][-1],
                      'R':data['R'][-1],
                      'I':data['I'][-1],
                      'A':data['A'][-1],
                      'H':data['H'][-1],
                      'D':data['D'][-1],
                      'S':S}

    return initial_values

def data_to_rates(data,params):
    
    #death rate of symptomatic
    r={}
    
    min_death_rate=0.001
    r['death_rate']=max((params['detection_rate']*data['D'][-1]/(data['I_cum'][-1]/(1+params['A_to_I_ratio'])),min_death_rate))
    r['I_to_D_prob']=r['Q_to_D_prob']=r['H_to_D_prob']=r['death_rate']

    #hosp rate of symptomatic
    min_hosp_rate=0.001
    r['hosp_rate']=max((params['detection_rate']*data['H_cum'][-1]/(data['I_cum'][-1]/(1+params['A_to_I_ratio'])),min_hosp_rate))
    r['I_to_H_prob']=r['Q_to_H_prob']=r['hosp_rate']

    return r

def doubling_time(case_data):
    days=len(case_data)
    Td=days/np.log2(case_data[-1]/case_data[0])
    return Td

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

