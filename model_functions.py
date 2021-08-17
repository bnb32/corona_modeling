import numpy as np
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
import re
from PIL import Image
import urllib.parse, urllib.request
from math import log, exp, tan, atan, pi, ceil
import matplotlib.pyplot as plt
import io
from scipy import interpolate
import progressbar

api_key='AIzaSyAP2OeLuXst4yOAfK8ayKoeslE4GXUeNsg'

def interp_mat(M,old_domain,new_domain,rows,cols):
    lons=np.linspace(old_domain[0],old_domain[1],M.shape[1])
    lats=np.linspace(old_domain[3],old_domain[2],M.shape[0])
    f=interpolate.interp2d(lons,lats,M,kind='cubic')
    lons_new=np.linspace(new_domain[0],new_domain[1],cols)
    lats_new=np.linspace(new_domain[3],new_domain[2],rows)
    return f(lons_new,lats_new)

def laplacian(M,dx,dy):
    rows,cols=M.shape
    L=np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            
            #interior
            if 1<=i<=(rows-2) and 1<=j<=(cols-2):
                L[i,j]=(M[i-1,j]+M[i+1,j]-2*M[i,j])/(dy*dy)+(M[i,j-1]+M[i,j+1]-2*M[i,j])/(dx*dx)
            
            # sides
            if i==0 and 1<=j<=(cols-2):
                L[i,j]=(M[i+1,j]-2*M[i,j])/(dy*dy)+(M[i,j-1]+M[i,j+1]-2*M[i,j])/(dx*dx)
            if i==rows-1 and 1<=j<=(cols-2):
                L[i,j]=(M[i-1,j]-2*M[i,j])/(dy*dy)+(M[i,j-1]+M[i,j+1]-2*M[i,j])/(dx*dx)
            if j==0 and 1<=i<=(rows-2):
                L[i,j]=(M[i-1,j]+M[i+1,j]-2*M[i,j])/(dy*dy)+(M[i,j+1]-2*M[i,j])/(dx*dx)
            if j==cols-1 and 1<=i<=(rows-2):
                L[i,j]=(M[i-1,j]+M[i+1,j]-2*M[i,j])/(dy*dy)+(M[i,j-1]-2*M[i,j])/(dx*dx)
            
            # corners
            if i==0 and j==0:
                L[i,j]=(M[i+1,j]-2*M[i,j])/(dy*dy)+(M[i,j+1]-2*M[i,j])/(dx*dx)
            if i==0 and j==cols-1:
                L[i,j]=(M[i+1,j]-2*M[i,j])/(dy*dy)+(M[i,j-1]-2*M[i,j])/(dx*dx)
            if i==rows-1 and j==0:
                L[i,j]=(M[i-1,j]-2*M[i,j])/(dy*dy)+(M[i,j+1]-2*M[i,j])/(dx*dx)
            if i==rows-1 and j==cols-1:
                L[i,j]=(M[i-1,j]-2*M[i,j])/(dy*dy)+(M[i,j-1]-2*M[i,j])/(dx*dx)
    return L

def sir_spatial_euler(S,I,R,P,beta,gamma,r0,dx,dy,dt):
    
    Sk1=-(I+(r0*r0/8.0)*laplacian(I,dx,dy))*(beta*S/P)*dt
    Rk1=gamma*I*dt
    Ik1=-(Sk1+Rk1)
    
    Sn=Sk1+S
    In=Ik1+I
    Rn=Rk1+R
    
    Sn[np.where(Sn<0)[0]]=0
    In[np.where(In<0)[0]]=0
    Rn[np.where(Rn<0)[0]]=0
    
    return Sn,In,Rn

def sir_spatial_rk(S,I,R,P,beta,gamma,r0,dx,dy,dt):
    
    #1st step    
    Sk1=-(I+(r0*r0/8.0)*laplacian(I,dx,dy))*(beta*S/P)*dt
    Rk1=gamma*I*dt
    Ik1=-(Sk1+Rk1)

    #2nd step
    Sn=Sk1/2+S
    In=Ik1/2+I
    Rn=Rk1/2+R
    
    Sn[np.where(Sn<0)[0]]=0
    In[np.where(In<0)[0]]=0
    
    Sk2=-(In+(r0*r0/8.0)*laplacian(In,dx,dy))*(beta*Sn/P)*dt
    Rk2=gamma*In*dt
    Ik2=-(Sk2+Rk2)
    
    #3rd step
    Sn=Sk2/2+S
    In=Ik2/2+I
    Rn=Rk2/2+R
    
    Sn[np.where(Sn<0)[0]]=0
    In[np.where(In<0)[0]]=0
    
    Sk3=-(In+(r0*r0/8.0)*laplacian(In,dx,dy))*(beta*Sn/P)*dt
    Rk3=gamma*In*dt
    Ik3=-(Sk3+Rk3)

    #3rd step
    Sn=Sk3*dt+S
    In=Ik3*dt+I
    Rn=Rk3*dt+R
    
    Sn[np.where(Sn<0)[0]]=0
    In[np.where(In<0)[0]]=0

    Sk4=-(In+(r0*r0/8.0)*laplacian(In,dx,dy))*(beta*Sn/P)*dt
    Rk4=gamma*In*dt
    Ik4=-(Sk4+Rk4)
    
    Sn=S+1/6*(Sk1+2*Sk2+2*Sk3+Sk4)
    In=I+1/6*(Ik1+2*Ik2+2*Ik3+Ik4)
    Rn=R+1/6*(Rk1+2*Rk2+2*Rk3+Rk4)

    Sn[np.where(Sn<0)[0]]=0
    In[np.where(In<0)[0]]=0
    Rn[np.where(Rn<0)[0]]=0
    
    return Sn,In,Rn   

def sir(S,I,R,beta,gamma,N):
    Sn=(-beta*S*I)+S
    In=(beta*S*I-gamma*I)+I
    Rn=gamma*I+R

    if Sn<0:
        Sn=0
    if In<0:
        In=0
    if Rn<0:
        Rn=0

    scale=N/(Sn+In+Rn)
    return Sn*scale,In*scale,Rn*scale

def seir(S,E,I,R,beta,gamma,a,N):
    Sn=(-beta*S*I)+S
    En=(beta*S*I-a*E)+E
    In=(a*E-gamma*I)+I
    Rn=gamma*I+R

    if Sn<0:
        Sn=0
    if En<0:
        En=0
    if In<0:
        In=0
    if Rn<0:
        Rn=0

    scale=N/(Sn+En+In+Rn)
    return Sn*scale,En*scale,In*scale,Rn*scale

def sim_sir(S,I,R,beta,gamma,n_days,beta_decay=None):
    N=S+I+R
    s,i,r=[S],[I],[R]
    for days in range(n_days):
        S,I,R=sir(S,I,R,beta,gamma,N)
        if beta_decay:
            beta*=(1-beta_decay)
        s.append(S)
        i.append(I)
        r.append(R)
    s,i,r=np.array(s),np.array(i),np.array(r)
    return s,i,r

def sim_seir(S,E,I,R,beta,gamma,a,n_days,beta_decay=None):
    N=S+E+I+R
    s,e,i,r=[S],[E],[I],[R]
    for days in range(n_days):
        S,E,I,R=seir(S,E,I,R,beta,gamma,a,N)
        if beta_decay:
            beta*=(1-beta_decay)
        s.append(S)
        e.append(E)
        i.append(I)
        r.append(R)
    s,e,i,r=np.array(s),np.array(e),np.array(i),np.array(r)
    return s,e,i,r

def sim_sir_spatial(S,I,R,P,beta,gamma,r0,dx,dy,dt,n_days,beta_decay=None):
    s,i,r=[S],[I],[R]
    with progressbar.ProgressBar(max_value=int(n_days/dt)) as bar:
        for t in range(int(n_days/dt)):
            bar.update(t)
            S,I,R=sir_spatial_rk(S,I,R,P,beta,gamma,r0,dx,dy,dt)
            #S,I,R=sir_spatial_euler(S,I,R,P,beta,gamma,r0,dx,dy,dt)
            if beta_decay:
                beta*=(1-beta_decay)
            s.append(S)
            i.append(I)
            r.append(R)
        s,i,r=np.array(s),np.array(i),np.array(r)
    return s,i,r

def get_colormap():

    # get colormap
    cmap=plt.cm.jet
    color_array = cmap(np.arange(cmap.N))
    
    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,cmap.N)
    
    # create a colormap object
    my_cmap = ListedColormap(color_array)
    
    return my_cmap

def get_map(domain):

    lon_min=domain[0]
    lon_max=domain[1]
    lat_min=domain[2]
    lat_max=domain[3]
    lat_mid=(lat_min+lat_max)/2
    lon_mid=(lon_min+lon_max)/2

    upperleft =  '%s,%s'%(lat_max,lon_min)
    lowerright = '%s,%s'%(lat_min,lon_max)
    
    zoom = latlonrangetozoom(lat_min,lat_max,lon_min,lon_max)   
    
    ullat, ullon = map(float, upperleft.split(','))
    lrlat, lrlon = map(float, lowerright.split(','))
    
    ulx, uly = latlontopixels(ullat, ullon, zoom)
    lrx, lry = latlontopixels(lrlat, lrlon, zoom)
    dx, dy = lrx - ulx, uly - lry
    position = '%s,%s'%(lat_mid,lon_mid)
    urlparams = urllib.parse.urlencode({'center': position,
                                  'zoom': str(zoom),
                                  'size': '%dx%d' %(dx,dy),
                                  'sensor': 'false',
                                  'key': api_key,
                                  'scale': 1})
    url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
    f=urllib.request.urlopen(url)
    im=Image.open(io.BytesIO(f.read()))
    
    return im


EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

def latlonrangetozoom(lat_min,lat_max,lon_min,lon_max):
    
    mx_min = (lon_min * ORIGIN_SHIFT) / 180.0
    my_min = log(tan((90 + lat_min) * pi/360.0))/(pi/180.0)
    my_min = (my_min * ORIGIN_SHIFT) /180.0
    
    mx_max = (lon_max * ORIGIN_SHIFT) / 180.0
    my_max = log(tan((90 + lat_max) * pi/360.0))/(pi/180.0)
    my_max = (my_max * ORIGIN_SHIFT) /180.0

    x_zoom = np.log2(640.0/(mx_max-mx_min)*INITIAL_RESOLUTION)
    y_zoom = np.log2(640.0/(my_max-my_min)*INITIAL_RESOLUTION)
    return np.min([int(x_zoom),int(y_zoom)])


def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

def pixelstolatlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon

def load_data(infile):
    f=open(infile,'r')
    lines=f.readlines()
    data=[]
    for l in lines:
        if "ncols" in l: 
            ncols=int(l.strip('ncols').strip(' '))
        elif "nrows" in l: 
            nrows=int(l.strip('nrows').strip(' '))
        elif "xllcorner" in l: 
            xllcorner=int(l.strip('xllcorner').strip(' '))
        elif "yllcorner" in l: 
            yllcorner=int(l.strip('yllcorner').strip(' '))
        elif "cellsize" in l: 
            cellsize=float(l.strip('cellsize').strip(' '))
        elif "NODATA_value" in l: 
            NODATA_value=int(l.strip('NODATA_value').strip(' '))
        else:
            data.append(list(l.split(' '))[0:-1])
    
    out=np.array(data,dtype=np.float)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i,j]==NODATA_value: out[i,j]=0
    return out, xllcorner, yllcorner, cellsize, NODATA_value
