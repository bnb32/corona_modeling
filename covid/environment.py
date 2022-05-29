"""Environment module"""
import os

COVID_DIR = os.path.dirname(os.path.realpath(__file__))

COVID_FIG_DIR = f'{COVID_DIR}/figs/'
LAMBDA_TASK_ROOT = os.environ['LAMBDA_TASK_ROOT'] = f'{COVID_DIR}/data/'

COVID_URI = "https: //covidtracking.com/api/states/daily?state = {0}"
COVID_ACT_API = ''
GOOGLE_API_KEY = ''
CSV_POP_FILE = '../../web/2018_state_county_pop.csv'
POPULATION_FILE = '../../web/2018_state_county_pop.json'
