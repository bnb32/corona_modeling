"""Environment module"""
import os

os.environ['COVID_REPO_DIR'] = '/home/ec2-user/environment/corona_modeling'
os.environ['COVID_FIG_DIR'] = f'{os.environ["COVID_REPO_DIR"]}/figs/'
os.environ['LAMBDA_TASK_ROOT'] = f'{os.environ["COVID_REPO_DIR"]}/data/'

COVID_URI = "https: //covidtracking.com/api/states/daily?state = {0}"
COVID_ACT_API = ''
GOOGLE_API_KEY = ''
CSV_POP_FILE = '../../web/2018_state_county_pop.csv'
POPULATION_FILE = '../../web/2018_state_county_pop.json'
