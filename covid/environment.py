import os

os.environ['COVID_REPO_DIR'] = '/home/ec2-user/environment/corona_modeling'
os.environ['COVID_FIG_DIR'] = f'{os.environ["COVID_REPO_DIR"]}/figs/'
os.environ['LAMBDA_TASK_ROOT'] = f'{os.environ["COVID_REPO_DIR"]}/data/'
covid_uri="https://covidtracking.com/api/states/daily?state={0}"

google_api_key=''
csv_pop_file='../../web/2018_state_county_pop.csv'
population_file='../../web/2018_state_county_pop.json'
