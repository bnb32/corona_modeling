import os

os.environ['LAMBDA_TASK_ROOT'] = '../data/'
covid_uri="https://covidtracking.com/api/states/daily?state={0}"

google_api_key=''
csv_pop_file='../../web/2018_state_county_pop.csv'
population_file='../../web/2018_state_county_pop.json'
