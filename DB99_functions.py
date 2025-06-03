#!/mnt/netapp2/Store_uni/home/usc/fp/aco/.conda/envs/venv_ms/bin/python
# coding: utf-8

# Import modules
import sys, os
import numpy as np 
import h5py
import datetime
import xarray as xr
from numba import jit

# --- FUNCTION read_trajdata
# The function read_trajdata reads the HDF5 dataset stored in the file given
# by filename, the only argument. In the case of FLEXPART-WRF simulations, trajectories
# are supposed to be already sorted, meaning that every element in the second axis of each
# array corresponds to one parcel.
def read_trajdata(filename):

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


# --- FUNCTION interpol_tcw_evap
# It interpolates total column and evaporation to the parcel positions
# It also interpolates total precipitation to the final parcel position
def interpol_tcw_evap_complete(data_path, release_date, x, y, dx, dy):

    ntimes, numpart = x.shape

    # Open the netCDF datasets for reading evaporation and precipitable water data
    tcw_ds = xr.open_dataset(data_path+'/tcw.nc', engine='netcdf4')
    evap_ds = xr.open_dataset(data_path+'/evap.nc', engine='netcdf4')

    # Handle times
    t_rel = datetime.datetime.strptime(release_date, '%Y%m%d%H')
    t_beg = t_rel-datetime.timedelta(hours=ntimes)

    # Take arrays
    evap_arr = evap_ds.sel(time=slice(t_beg, t_rel)).e.values
    evap_arr = np.where(evap_arr>0, 0.0, -evap_arr)
    tcw_arr = tcw_ds.sel(time=slice(t_beg, t_rel)).tcw.values
    tcw_ds.close()
    evap_ds.close()

    tcw, evap = compute_tcw_evap(x, y, tcw_arr, evap_arr, dx, dy)
    tcw = np.where(x==-1000.0, 0.0, tcw)
    evap = np.where(x==-1000.0, 0.0, evap)

    # Finally, handle precipitation data
    x0_ind = np.round(x[0,:]/dx).astype(np.int32)
    y0_ind = np.round((-y[0,:]+90.0)/dy).astype(np.int32)
    tp_ds = xr.open_dataset(data_path+'/rain.nc', engine='netcdf4')
    tp_ds_arr = tp_ds.sel(Time=t_rel).rain.values
    tp = tp_ds_arr[y0_ind, x0_ind]

    return tcw, evap, tp

# --- FUNCTION compute_tcw_evap
# Adaptation of the previous function (interpol_tcw_evap) to be performed with numba
# Recall that ERA5 data is given in the box 90>=lat>=-90, 0<=lon<=360
@jit(nopython=True)
def compute_tcw_evap(x, y, tcw_arr, evap_arr, dx, dy):

    ntimes, numpart = x.shape
    nx = int(360/dx)
    ny = int(180/dy)+1

    x_ind = np.round_(x/dx).astype(np.int32)
    y_ind = np.round_((-y+90.0)/dy).astype(np.int32)
    tcw = np.zeros((ntimes, numpart))
    evap = np.zeros((ntimes, numpart))

    for t in range(ntimes):
        for j in range(numpart):
            tcw[t,j] = tcw_arr[-1-t, y_ind[t,j], x_ind[t,j]]
            evap[t,j] = evap_arr[-1-t, y_ind[t,j], x_ind[t,j]]

    return tcw, evap


# --- FUNCTION compute_endpoints
# It computes the indices associated to the end point on the surface of the Earth (height is not considered)
def compute_endpoints(x, y, dx, dy):

    nx = int(360/dx)
    ny = int(180/dy)+1

    x_ind = np.round_((x[:-1,:]+180.0)/dx).astype(int)
    x_ind[x_ind==nx] = 0
    x_ind = np.where(x[1:,:]==-1000.0, -1000, x_ind)
    y_ind = np.round_((y[:-1,:]+90.0)/dy).astype(int)
    y_ind = np.where(x[1:,:]==-1000.0, -1000, y_ind)
    
    return x_ind, y_ind


# --- FUNCTION discount_db99
# Given the mid point parcel positions in x_ind and y_ind (relative to
# the grid we are using), together with the precipitable water and evaporation
# in the same grid cells, the backwards DB99 method is applied (such as in the
# UTRACK methodology), to compute the contributions for each parcel and time.
# The array with shape ntimes x numpart f with these contributions is returned
def discount_db99(tcw, evap):

    ntimes, numpart = tcw.shape 

    # Indices for evaporation
    ind_evap = evap[:-1,:]>0
    tcw[tcw==0.0] = np.nan
    fr = np.where(ind_evap==True, np.nan_to_num(997.0*evap[:-1,:]/tcw[:-1,:]), 0)

    # Water mass parcels already have at the trajectory beginning and
    # initialization of the moisture contribution proportion
    f = np.zeros((ntimes,numpart))       
    f[0,:] = 1.
    
    # Loop over times previous to precipitation
    # When we start at step 0 we are computing the moisture sources for all the 
    # water in the atmospheric column in the last time
    for t in range(0, ntimes-1, 1):
        #
        # Select parcels gaining moisture at this time, and attribute
        ind_gain = ind_evap[t,:]
        frr = fr[t, ind_gain]
        frr = frr/(1+frr)
        f[t+1,ind_gain] = f[0,ind_gain]*frr 
        f[0,ind_gain] = f[0,ind_gain]*(1-frr)

    return f


# ---FUNCTION compute_ms_field
# Given the mid point parcel positions in x_ind and y_ind, and given the 
# contributuons from each parcel and time f, it computes the spatial distribution
# of moisture sources in the grid we are using.
# For FLEXPART-WRF simulations, if x or y is 1000, the parcel leaves the domain through
# the south or north border. We use the previous time step to determine which border it crossed.
@jit(nopython=True)
def compute_ms_field(x_ind, y_ind, f, dx, dy):

    ntimes, numpart = f.shape
    nx = int(360/dx)
    ny = int(180/dy)+1
    ms = np.zeros((nx,ny))
    ms_north = 0.
    ms_south = 0.
    
    # Guess if this event is in the NH or SH to distinguish between North and South border
    if y_ind[0,:].mean()>ny/2:
        ny_border = int(3*(ny-1)/4)
    else:
        ny_border = int(1*(ny-1)/4)

    for j in range(numpart):
        for t in range(1, ntimes, 1):
            if x_ind[t-1,j]>=0:
                ms[x_ind[t-1,j], y_ind[t-1,j]] = ms[x_ind[t-1,j], y_ind[t-1,j]] + f[t,j]
            else:
                if y_ind[t-2, j]>ny_border:
                    ms_north = ms_north+f[0,j]
                else:
                    ms_south = ms_south+f[0,j]

    return ms, ms_north, ms_south



# --- FUNCTION get_results
# It writes the 2D spatial distribution in kg and mm, together with the 3D contributions from
# the north and south boundaries, to a netcdf dataset
def get_results(release_date, ms, ms_north, ms_south):

    # Take the resolution from the shape of the 2D distribution
    # Define latitudes and longitudes
    nx, ny = ms.shape
    dx = float(360/nx)
    dy = float(180/(ny-1))
    lats = np.linspace(-90, 90, ny)
    lons = np.linspace(-180, 180-dx, nx)

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
