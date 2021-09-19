import numpy as np
import matplotlib.pyplot as plt
import os
from models import sim_sir_spatial

# initial conditions
total_population=100
S=total_population
initial_infections=10
doubling_time=3
current_hosp=5
hosp_market_share=0.15
hosp_rate=0.05
total_infections=current_hosp/hosp_market_share/hosp_rate
detection_prob=initial_infections/total_infections
intrinsic_growth_rate=2**(1/doubling_time)-1
recovery_days=14.0
gamma=1/recovery_days
relative_contact_rate=0.0
beta=(intrinsic_growth_rate+gamma)/S*(1-relative_contact_rate)
r_t=beta/gamma*S
r_naught=r_t/(1-relative_contact_rate)
beta_decay=0.0
S,I,R=S,initial_infections/detection_prob,0

Xsize=1.0
Ysize=1.0
dr=0.01
dt=0.25
dimx=int(Xsize/dr)
dimy=int(Ysize/dr)
n_days=50
r0=0.02
beta=2
gamma=0.1
a=0.1

S=np.zeros((dimx,dimy))
I=np.zeros((dimx,dimy))
R=np.zeros((dimx,dimy))
P=np.zeros((dimx,dimy))

#initial distributions
def I0(i,j,a,dr):
    r=np.sqrt((i*dr-0.5)**2+(j*dr-0.5)**2)
    tmp=32.0/(3.14*a**4)*(a*a/4-r*r)
    if r<=a/2: return tmp
    else: return 0

def Pop(i,j,dr):
    return 15000-10000*(i*dr)

for i in range(dimx):
    for j in range(dimy):
        I[i,j]=I0(i,j,a,dr)
        P[i,j]=Pop(i,j,dr)
        S[i,j]=P[i,j]-I[i,j]

s,i,r=sim_sir_spatial(S,I,R,P,beta,gamma,r0,dr,dr,dt,n_days,beta_decay=beta_decay)

cmap=plt.cm.jet
os.system('rm -f ./gif/*.png')
for t in range(int(n_days/dt)):
    plt.clf()
    plt.imshow(i[t],cmap=cmap,interpolation='bilinear',aspect='auto')
    plt.savefig('./gif/infected_%s.png'%(format(t,'04')))

os.system('convert -delay 20 -loop 0 ./gif/infected_*.png ./gif/infected.gif')

plt.clf()
for t in range(0,int(n_days/dt),int(5/dt)):
    plt.plot(i[t][:,int(dimy/2)])
plt.savefig('./infected.png')
