from covid.postprocessing import load_data,latlontopixels,pixelstolatlon,get_map,get_colormap,interp_pop

from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import urllib.parse, urllib.request
import io
from math import log, exp, tan, atan, pi, ceil
import matplotlib.pyplot as plt
import numpy as np

#data file
infile='./data/gpw_v4_population_count_rev11_2020_2pt5_min.asc'
#infile='./data/usap00ag.asc'
P,lon0,lat0,dr,no_val=load_data(infile)

lat_min=42.0
lat_max=43.0
lon_min=-77.0
lon_max=-76.0

get_colormap()

#cmap=plt.cm.jet
im=get_map(lat_min,lat_max,lon_min,lon_max)
Pnew=interp_pop(P,lat_min,lat_max,lon_min,lon_max,im.size[0],im.size[1])
print("image size: %s, %s"%(im.size))

plt.imshow(im,extent=[lon_min,lon_max,lat_min,lat_max])
plt.imshow(Pnew,cmap='rainbow_alpha',extent=[lon_min,lon_max,lat_min,lat_max],origin='lower')
plt.savefig('test_fig.png')

