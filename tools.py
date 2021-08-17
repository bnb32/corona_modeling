import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

def model_parameters(S,I,R):
    recovery_time=14.0
    doubling_time=3.0
    gamma=1/recovery_time
    g=2**(1/doubling_time)-1
    isolation_percent=0.1
    beta=(g+gamma)*(S+I+R)/S*(1-isolation_percent)
    n_days=100
    return beta,gamma,n_days


def location_to_initial_values(lat,lon):
    N=1000
    I=1
    R=0
    S=N-I-R
    
    return S,I,R

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

def plot_infected(I):
    
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

    plt.savefig('test.png')
    
