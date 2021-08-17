import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
import urllib.request
import json
import environment as env

api_key=env.api_key
population_file=env.population_file
json_file=open(population_file)
pop_data=json.load(json_file)

def location_to_cases(state):
    url="https://covidtracking.com/api/states?state=%s"%(state)
    v=urllib.request.urlopen(url).read()
    j=json.loads(v)
    return j["positive"]

def location_to_ts(state):
    url="https://covidtracking.com/api/states/daily?state=%s"%(state)
    v=urllib.request.urlopen(url).read()
    j=json.loads(v)
    return j

def location_to_doubling_time(state):
    j=location_to_ts(state)
    f_date=str(j[1]["date"])
    l_date=str(j[0]["date"])
    f_date=date(int(f_date[0:4]),int(f_date[4:6]),int(f_date[6:]))
    l_date=date(int(l_date[0:4]),int(l_date[4:6]),int(l_date[6:]))
    delta=l_date-f_date
    days=delta.days
    
    Td=days/np.log2(j[0]["positive"]/j[1]["positive"])
    return Td

def location_to_population(state,county=None):
    if county==None: county=state
    N=pop_data[state][county]    
    return N

def model_parameters(S,I,R,Tr=14.0,Td=6.0,Sd=0.0):
    gamma=1/Tr
    g=2**(1/Td)-1
    N=S+I+R
    beta=(g+gamma)*N/S*(1-Sd)
    return beta,gamma

def latlon_to_place(lat,lon):
    url="https://maps.googleapis.com/maps/api/geocode/json?"
    url+="latlng=%s,%s&sensor=false&key=%s" %(lat,lon,api_key)
    v=urllib.request.urlopen(url).read()
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

def sir(S,I,R,beta,gamma,N):
    Sn=(-beta*S*I)/N+S
    In=(beta*S*I/N-gamma*I)+I
    Rn=gamma*I+R

    if Sn<0:
        Sn=0
    if In<0:
        In=0
    if Rn<0:
        Rn=0

    return Sn,In,Rn

def sim_sir(S,I,R,beta,gamma,n_days):
    N=S+I+R
    s,i,r=[S],[I],[R]
    for days in range(n_days):
        S,I,R=sir(S,I,R,beta,gamma,N)
        s.append(S)
        i.append(I)
        r.append(R)
    s,i,r=np.array(s),np.array(i),np.array(r)
    return s,i,r

def plot_infected(I,state,county=None):
    
    if county==None:
        filename='../../web/'+state.replace(' ','')+'_infected.png'
    else:
        filename='../../web/'+state.replace(' ','')+'_'+county.replace(' ','')+'_infected.png'
    dates=[]
    for i in range(len(I)):
        day=str((date.today()+timedelta(days=i)).strftime('%m-%d'))
        dates+=[day]
    plt.plot(dates,I)
    ax=plt.gca()
    plt.xticks(rotation=60)
    
    for i,label in enumerate(ax.get_xticklabels()): 
        if i % int(len(I)/10):
            label.set_visible(False)
    
    plt.ylabel("Number Infected")
    plt.xlabel("Date")
    if county==None:
        plt.title("%s"%(state))
    else:    
        plt.title("%s-%s"%(state,county))

    plt.savefig(filename)
