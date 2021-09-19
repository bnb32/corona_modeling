from covid.postprocessing import load_data,get_colormap,get_map
from covid.models import sim_sir_spatial
from covid.misc import interp_mat

import numpy as np
import matplotlib.pyplot as plt
import urllib.request as urllib2
from PIL import Image
import os
import io
import argparse

#data file
infile='./data/gpw_v4_population_count_rev11_2020_2pt5_min.asc'
pop_array_file='./data/pop_array.npy'
pop_interp_array_file='./data/pop_interp_array.npy'
gm_map_file="./data/map_data.png"

parser=argparse.ArgumentParser(description="Run Spatial SIR Model")
parser.add_argument('-load_raw_file',default=False,action='store_true')
parser.add_argument('-load_raw_array',default=False,action='store_true')
parser.add_argument('-save_raw_array',default=False,action='store_true')
parser.add_argument('-interp_raw',default=False,action='store_true')
parser.add_argument('-save_interp',default=False,action='store_true')
parser.add_argument('-load_interp',default=False,action='store_true')
parser.add_argument('-request_map',default=False,action='store_true')
parser.add_argument('-load_map',default=False,action='store_true')
parser.add_argument('-save_map',default=False,action='store_true')
args=parser.parse_args()

#domain
lat_min=42.0
lat_max=43.0
lat_range=np.abs(lat_max-lat_min)
lon_min=-77.0
lon_max=-76.0
lon_range=np.abs(lon_max-lon_min)
cols=100
rows=100
old_domain=[-180.0,180.0,-90.0,90.0]
new_domain=[lon_min,lon_max,lat_min,lat_max]

#load raw data
if args.load_raw_file:
    print("Loading raw data")
    P,lon0,lat0,dr,no_val=load_data(infile)

if args.save_raw_array:    
    print("Saving raw data")
    np.save(pop_array_file,P)

if args.load_raw_array:
    print("Loading raw array")
    P=np.load(pop_array_file)

#get google map
cmap=get_colormap()

if args.request_map:
    im=get_map(new_domain)

if args.save_map:
    im.save(gm_map_file,extend=new_domain,origin='lower')

if args.load_map:
    im=Image.open(gm_map_file)

im_rows=im.size[1]
im_cols=im.size[0]

#interpolate population to map grid
if args.interp_raw:
    print("Interpolating data")
    P=interp_mat(P,old_domain,new_domain,rows,cols)

if args.save_interp:
    print("Saving interpolated data")
    np.save(pop_interp_array_file,P)

if args.load_interp:
    print("Loading interpolated data")
    P=np.load(pop_interp_array_file)

#grid dimensions
dx=float(lon_range/cols)
dy=float(lat_range/rows)
dr=np.sqrt(dx*dx+dy*dy)

#multiply population vals by area
for i in range(rows):
    for j in range(cols):
        if P[i,j]>0:
            P[i,j]=P[i,j]*25


#problems with zero population regions
P+=1

# initial conditions
dt=0.25
n_days=60
doubling_time=3*dt
intrinsic_growth_rate=2**(1/doubling_time)-1
recovery_time=20*dt
gamma=1/recovery_time
relative_contact_rate=0.0
beta=(intrinsic_growth_rate+gamma)*(1-relative_contact_rate)
beta_decay=0.0
r0=2*dr#100*dr
a=10*dr

print("Simulation values")
print("dt: %s" %dt)
print("ndays: %s" %n_days)
print("time steps: %s" %(int(n_days/dt)))
print("beta: %s" %beta)
print("gamma: %s" %gamma)
print("interaction radius: %s" %r0)
print("initial infection radius: %s" %a)

S=np.zeros((rows,cols))
I=np.zeros((rows,cols))
R=np.zeros((rows,cols))

#initial distributions
def I0(i,j,a,dr):
    r=np.sqrt(dx*dx*(i-cols/2)**2+dy*dy*(j-rows/2)**2)
    tmp=32.0/(3.14*a**4)*(a*a/4.0-r*r)
    if r<=a/2: return tmp
    else: return 0

for i in range(rows):
    for j in range(cols):
        I[i,j]=I0(j,i,a,dr)
        S[i,j]=P[i,j]-I[i,j]

#initial plots
print("Plotting initial distributions")
plt.imshow(im,extent=new_domain)
plt.imshow(interp_mat(P,new_domain,new_domain,im_rows,im_cols),cmap=cmap,aspect='auto',extent=new_domain)
plt.savefig('./population.png')

plt.clf()
plt.imshow(im,extent=new_domain)
plt.imshow(interp_mat(I,new_domain,new_domain,im_rows,im_cols),cmap=cmap,aspect='auto',extent=new_domain)
plt.savefig('./initial_infection.png')

plt.clf()
plt.imshow(im,extent=new_domain)
plt.imshow(interp_mat(S,new_domain,new_domain,im_rows,im_cols),cmap=cmap,aspect='auto',extent=new_domain)
plt.savefig('./initial_susceptible.png')

#run simulation
print("running simulation")
s,i,r=sim_sir_spatial(S,I,R,P,beta,gamma,r0,dx,dy,dt,n_days,beta_decay=beta_decay)


#make gif
print("Making infected gif")
os.system('rm -f ./gif/*.png')
for t in range(int(n_days/dt)):
    plt.clf()
    plt.imshow(im,extent=new_domain)
    plt.imshow(interp_mat(i[t],new_domain,new_domain,im_rows,im_cols),cmap=cmap,aspect='auto',extent=new_domain)
    plt.savefig('./gif/infected_%s.png'%(format(t,'04')))

os.system('convert -delay 20 -loop 0 ./gif/infected_*.png ./gif/infected.gif')

print("Making susceptible gif")
for t in range(int(n_days/dt)):
    plt.clf()
    plt.imshow(im,extent=new_domain)
    plt.imshow(interp_mat(s[t],new_domain,new_domain,im_rows,im_cols),cmap=cmap,aspect='auto',extent=new_domain)
    plt.savefig('./gif/susceptible_%s.png'%(format(t,'04')))

os.system('convert -delay 20 -loop 0 ./gif/susceptible_*.png ./gif/susceptible.gif')

