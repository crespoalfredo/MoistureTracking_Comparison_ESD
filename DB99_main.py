#!/mnt/netapp2/Store_uni/home/usc/fp/aco/.conda/envs/venv_ms/bin/python
# coding: utf-8

# Import modules
import sys, os
import numpy as np 
import h5py
import xarray as xr
import datetime
from numba import jit
import configparser
from DB99_functions import *

try:
    namelist = sys.argv[1]
except:
    namelist = 'namelists_SOD08/namelist_NoABL'


# Command line arguments
config = configparser.ConfigParser()
config.read(namelist)
release_date = config['DATES']['release_date']
model = config['OPTIONS']['model']
dt = int(config['OPTIONS']['dt'])
rh_prec = float(config['OPTIONS']['rh_prec'])
z_prec = float(config['OPTIONS']['z_prec'])
dx = float(config['OPTIONS']['dx'])
dy = float(config['OPTIONS']['dy'])
path_traj = config['PATHS']['path_traj']
path_data = config['PATHS']['path_otherdata']
path_out = config['PATHS']['path_out']
prefix_traj = config['PATHS']['prefix_traj']

# Read the HDF5 dataset with trajectories from FLEXPART
filename = path_traj+prefix_traj+release_date+'.h5'
x, y, z, h_abl, q, RH, mass = read_trajdata(filename)

# Restrict to the desired times
times = np.arange(0, x.shape[0], dt)
x = x[times,:]; y = y[times,:]
q = q[times,:]; RH = RH[times,:]
z = z[times,:]; h_abl = h_abl[times,:]

# Compute the indices corresponding to the mid point parcel positions
x_ind, y_ind = compute_endpoints(x, y, dx, dy)

# Interpolate total column water and evaporation to parcel position, precipitation only for the last time step
tcw, evap, tp = interpol_tcw_evap_complete(path_data, release_date, x, y, dx, dy)

# Select parcels contributing to precipitation
ind_rain = np.where((RH[0,:]>rh_prec)&(z[0,:]>z_prec))[0]
x = x[:, ind_rain]
x_ind = x_ind[:, ind_rain]; y_ind = y_ind[:, ind_rain]
q = q[:, ind_rain]; RH = RH[:, ind_rain]
z = z[:, ind_rain]; h_abl = h_abl[:, ind_rain]
tcw = tcw[:, ind_rain]; evap = evap[:, ind_rain]; tp = tp[ind_rain]
ntimes, numpart = q.shape

# Discounting routine
f = q*mass*discount_db99(tcw, evap)

# Consider precipitation and precipitable water per grid cell (precipitation is assumed to be in mm)
f = f*997.0*tp/(tcw[0,:]*1000.0)

# Now compute the spatial distribution of the preciptiation sources
ms, ms_north, ms_south = compute_ms_field(x_ind, y_ind, f, dx, dy)

# Finally, write the results to a netcdf dataset
ds = get_results(release_date, ms, ms_north, ms_south)
z_prec = str(int(z_prec)).replace('.', '')
rh_prec = str(int(rh_prec)).replace('.', '')
filename_out = path_out+'ms_db99-{}_z0{}-rh{}_{}.nc'.format(model, z_prec, rh_prec, release_date)
ds.to_netcdf(filename_out)