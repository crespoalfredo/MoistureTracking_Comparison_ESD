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
from SOD08_functions import *

try:
    namelist = sys.argv[1]
except:
    namelist = 'namelist'

# Command line arguments
config = configparser.ConfigParser()
config.read(namelist)
release_date = config['DATES']['release_date']
model = config['OPTIONS']['model']
dt = int(config['OPTIONS']['dt'])
dq_evap = float(config['OPTIONS']['dq_evap'])
dq_prec = float(config['OPTIONS']['dq_prec'])
rh_prec = float(config['OPTIONS']['rh_prec'])
rh_disc = float(config['OPTIONS']['rh_disc'])
dx = float(config['OPTIONS']['dx'])
dy = float(config['OPTIONS']['dy'])
abl = int(config['OPTIONS']['abl'])
abl_factor = float(config['OPTIONS']['abl_factor'])
path_traj = config['PATHS']['path_traj']
path_data = config['PATHS']['path_otherdata']
path_out = config['PATHS']['path_out']
prefix_traj = config['PATHS']['prefix_traj']

# Read the HDF5 dataset with trajectories from FLEXPART
filename = path_traj+prefix_traj+release_date+'.h5'
x, y, z, h_abl, q, RH, mass = read_trajdata_sodemann_reduced_abl(filename)

# Restrict to the desired times
times = np.arange(0, x.shape[0], dt)
x = x[times,:]; y = y[times,:]
q = q[times,:]; RH = RH[times,:]
z = z[times,:]; h_abl = h_abl[times,:]

# Compute the indices corresponding to the mid point parcel positions
x_ind, y_ind = compute_midpoints(x, y, dx, dy)

# Select parcels contributing to precipitation
dq_end = q[1,:]-q[0,:]
RH_mean = 0.5*(RH[0,:]+RH[1,:])
ind_rain = np.where((RH_mean>rh_prec)&(dq_end>dq_prec))[0]
x = x[:, ind_rain]
x_ind = x_ind[:, ind_rain]; y_ind = y_ind[:, ind_rain]
q = q[:, ind_rain]; RH = RH[:, ind_rain]
z = z[:, ind_rain]; h_abl = h_abl[:, ind_rain]
ntimes, numpart = q.shape

# Find those times where the parcel gains and loses water
dq = q[:-1,:]-q[1:,:]
z_mean = 0.5*z[:-1,:]+0.5*z[1:,:]
h_abl_mean = 0.5*h_abl[:-1,:]+0.5*h_abl[1:,:]
ind_prec = (dq<dq_prec)&(RH[1:,:]>rh_disc)
if abl==1:
    ind_evap = (dq>dq_evap*0.001)&(z_mean<abl_factor*h_abl_mean)
else:
    ind_evap = (dq>dq_evap*0.001)

# Correct dq when trajectories come from FLEXPART-WRF
if model=='WRF':
    time_correct = (x==-1000.0).argmax(axis=0)
    part_correct = np.where(time_correct>0)[0]
    time_correct = time_correct[part_correct]
    dq[time_correct-1, part_correct] = q[time_correct-1, part_correct]

# Discounting routine
f = discount_sod08(dq, q, ind_evap, ind_prec)

# Upscale to the total humidity before precipitation occurs
f = f*q[1,:].sum()*mass/f.sum()

# Take the fraction represented by the water lost in the last step
f = -f*dq[0,:]/q[1,:]

# Now compute the spatial distribution of the preciptiation sources
ms, ms_north, ms_south = compute_ms_field(x_ind, y_ind, f, dx, dy)

# Finally, write the results to a netcdf dataset
ds = get_results(release_date, ms, ms_north, ms_south)
dq_evap = str(dq_evap).replace('.', '')
rh_disc = str(int(rh_disc)).replace('.', '')
if abl==1:
    filename_out = path_out+'ms_sod08ABL_dt{}-dq{}-rh{}_{}.nc'.format(dt, dq_evap, rh_disc, release_date)
else:
    filename_out = path_out+'ms_sod08_dt{}-dq{}-rh{}_{}.nc'.format(dt, dq_evap, rh_disc, release_date)    

ds.to_netcdf(filename_out)