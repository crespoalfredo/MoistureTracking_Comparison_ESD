#!/mnt/netapp2/Store_uni/home/usc/fp/aco/.conda/envs/venv_ms/bin/python
# coding: utf-8

# Import modules
import sys, os
import numpy as np 
import h5py
import datetime
import xarray as xr
from numba import jit

# --- FUNCTION read_trajdata_complete
# The function read_trajdata reads the HDF5 dataset stored in the file given
# by filename, the only argument. In the case of FLEXPART-WRF simulations, trajectories
# are supposed to be already sorted, meaning that every element in the second axis of each
# array corresponds to one parcel.
def read_trajdata_sodemann_reduced_abl(filename):

    # Open the HDF5 dataset for reading
    with h5py.File(filename, 'r') as f:
        dset = f['trajdata']
        x = dset[:,:,0]                    # longitude in degrees
        x[x>180.0] = x[x>180.0]-360.0
        y = dset[:,:,1]                    # latitude in degrees
        z = dset[:,:,2]                    # height in m
        h_abl = dset[:,:,3]                # PBL height in m
        q = dset[:,:,4]                    # specific humidity in kg kg-1
        RH = dset[:,:,5]                   # relative humidity
        mass = f['other_data'][:][0]       # average parcel mass

    return x, y, z, h_abl, q, RH, mass


# --- FUNCTION compute_midpoints
# It computes the mid point on the surface of the Earth (height is not considered)
def compute_midpoints(x, y, dx, dy):

    nx = int(360/dx)
    ny = int(180/dy)+1
    
    xmidp = 0.5*np.cos(y[:-1,:]*np.pi/180.0)*np.cos(x[:-1,:]*np.pi/180.0)
    xmidp = xmidp + 0.5*np.cos(y[1:,:]*np.pi/180.0)*np.cos(x[1:,:]*np.pi/180.0)
    ymidp = 0.5*np.cos(y[:-1,:]*np.pi/180.0)*np.sin(x[:-1,:]*np.pi/180.0)
    ymidp = ymidp + 0.5*np.cos(y[1:,:]*np.pi/180.0)*np.sin(x[1:,:]*np.pi/180.0)
    zmidp = 0.5*(np.sin(y[:-1,:]*np.pi/180.0)+np.sin(y[1:,:]*np.pi/180.0))
    latmidp = np.arctan2(zmidp, np.sqrt(xmidp*xmidp+ymidp*ymidp))*180.0/np.pi
    lonmidp = np.arctan2(ymidp, xmidp)*180.0/np.pi

    x_ind = np.round_((lonmidp+180.0)/dx).astype(int)
    x_ind[x_ind==nx] = 0
    x_ind = np.where(x[1:,:]==-1000.0, -1000, x_ind)
    y_ind = np.round_((latmidp+90.0)/dy).astype(int)
    y_ind = np.where(x[1:,:]==-1000.0, -1000, y_ind)
    
    return x_ind, y_ind


# --- FUNCTION discount_sod08
# The linear discounting from Sodemann et al., (2008) is used to compute the contributions for each
# parcel and time.
# The contribution is given by the relative increment in humidity.
# The array with shape ntimes x numpart f with these contributions is returned.
@jit(nopython=True)
def discount_sod08(dq, q, ind_evap, ind_prec):

    ntimes, numpart = q.shape

    f =  np.zeros((ntimes, numpart))
    f[0,:] = np.where(q[-1,:]>0, q[-1,:], 0.0)

    for t in range(1, ntimes-1, 1):

        ind_gain = ind_evap[-t, :]==True
        f[t, ind_gain] = dq[-t, ind_gain]
        ind_loss = ind_prec[-t,:]==True
        f[t,ind_loss] = 0
        qfactor = np.where(ind_loss==True, q[-(t+1),:]/q[-t,:], 1)

        for s in range(t):
            f_old = f[s,:]
            f[s,:] = f_old*qfactor

    return f


# ---FUNCTION compute_ms_field
# Given the mid point parcel positions in x_ind and y_ind, and given the 
# contributuons from each parcel and time f, it computes the spatial distribution
# of moisture sources in the grid we are using
@jit(nopython=True)
def compute_ms_field(x_ind, y_ind, f, dx, dy):

    ntimes, numpart = f.shape

    nx = int(360/dx)
    ny = int(180/dy)+1

    ms = np.zeros((nx,ny))
    ms_north = 0.
    ms_south = 0.
    ny34 = int(3*(ny-1)/4)
    for j in range(numpart):
        for t in range(1, ntimes, 1):
            if x_ind[-t,j]>=0:
                ms[x_ind[-t,j], y_ind[-t,j]] = ms[x_ind[-t,j], y_ind[-t,j]] + f[t-1,j]
            else:
                if y_ind[t-2,j]>ny34:
                    ms_north = ms_north+f[-1,j]
                else:
                    ms_south = ms_south+f[-1,j]

    return ms, ms_north, ms_south



# --- FUNCTION get_results
# It writes the 2D spatial distribution in kg and mm, together with the 3D contributions from
# the north and south boundaries, to a netcdf dataset
def get_results(release_date, ms, ms_north, ms_south):

    # Take the resolution from the shape of the 2D distribution
    nx, ny = ms.shape
    dx = float(360/nx)
    dy = float(180/(ny-1))
    lats = np.linspace(-90, 90, ny)
    lons = np.linspace(-180, 180-dx, nx)

    # Define latitudes and longitudes

    # Now transpose the 2D array and reshape the 3D north and south contributions
    ms = np.transpose(ms)
    ms = np.reshape(ms, (1,)+ms.shape)
    ms_north = np.reshape(ms_north, (1,))
    ms_south = np.reshape(ms_south, (1,))

    # Convert to mm 
    lats_arr = np.repeat(lats, lons.shape).reshape(ms[0,:,:].shape)
    area_arr = np.sin((lats_arr+0.5*dy)*np.pi/180.0)-np.sin((lats_arr-0.5*dy)*np.pi/180.0)
    area_arr = dx*np.pi/180.0*6371.0*1000.0*6371.0*1000.0*area_arr
    area_arr[area_arr==0.0] = np.NaN
    inv_density = 1000.0/997.0
    ms_mm = np.nan_to_num(ms[0,:,:]*inv_density/area_arr)
    ms_mm = np.reshape(ms_mm, ms.shape)

    # Handle time
    rel_date = datetime.datetime.strptime(release_date, '%Y%m%d%H')
    rel_date = [rel_date.strftime('%Y-%m-%dT%H:%M')]
    rel_date = np.array(rel_date, dtype='datetime64[ns]')

    # Create and store the dataset
    ds = xr.Dataset(data_vars=dict(ms_2d=(["time", "lat", "lon"], ms),
                               ms_mm=(["time", "lat", "lon"], ms_mm),
                               ms_3d_south=("time", ms_south),
                               ms_3d_north=("time", ms_north)),
                coords=dict(time=("time", rel_date), lat=("lat", lats), lon=("lon", lons)))

    return ds
