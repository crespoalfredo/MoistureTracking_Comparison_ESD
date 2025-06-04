#!/mnt/netapp2/Store_uni/home/usc/fp/aco/.conda/envs/ENV/bin/python
# coding: utf-8

# Import modules
import sys
import os
import glob
import numpy as np
import struct
import h5py
import re
import xarray as xr
from datetime import datetime
from multiprocessing import Pool

# --- FUNCTION readpart10 (from FLEXPART website)
# It reads FLEXPART binary output and converts it to a numpy array
def readpart10(directory, i, numpart, nspec=1):

    # determine path for the file to read
    if isinstance(i, (int, float)):
        dates_file = open(directory+'/dates', 'r')
        dates = dates_file.read()
        dates_file.close()
        dates = dates.splitlines()
        file = 'partposit_'+str(dates[i])
    elif isinstance(i,(str)):
        file = 'partposit_'+i
    file = directory+'/'+file

    # if numpart not given, calculate numpart (total) and return it
    if isinstance(numpart, (str)) :
        # the following assumes that the file has the exact format written in partouput.f90 of Flexppart v10.4
        # then the size of the file is exactly 4 * [3+(numpart+1)*(14+nspec)] bytes
        nbytes = os.path.getsize(file)
        numpart = round((nbytes/4+3)/(14+nspec)-1);
        output = numpart
        print(int(numpart))
        return
    #
    # initialize arrays to hold return values
    outheader = 0
    npoint = np.zeros((numpart, 1), dtype = np.int32)       # particle id
    xyz = np.zeros((numpart, 3), dtype = np.float32)        # longitude, latitude and elevation (with topography)
    itramem = np.zeros((numpart, 1), dtype = np.int32)      # memorized particle release time
    vars = np.zeros((numpart, 7), dtype = np.float32)       # topo: topography
#                                                          # pvi: potential vorticity
#                                                          # qvi: specific humidity
#                                                          # rhoi: density, converted to pressure
#                                                          # hmixi: PBL height
#                                                          # tri: tropopause height
#                                                          # tti: temperature [K]
    xmass = np.zeros((numpart, nspec), dtype = np.float32)  # mass of the species along the trajectory

    #  read file header (time stamp)
    try :
        particle_file = open(file, 'rb')
    except OSError:
        print('Could not open/read file:', file)
        sys.exit()

    dummy = particle_file.read(4) # read first dummy value
    packed_outheader = particle_file.read(4)
    outheader = struct.unpack('@i',packed_outheader)
    dummy = particle_file.read(4) # read next dummy value

    particle_file.close()

    # check that the number of particles is at least 1
    if numpart<1 :
        print('ERROR : number of particles must be at least 1. \n')
        return
    #
    # note : FLEXPART writes the output as
    #         write(unitpartout) npoint(i),xlon,ylat,ztra1(i),
    #    +    itramem(i),topo,pvi,qvi,rhoi,hmixi,tri,tti,
    #    +    (xmass1(i,j),j=1,nspec)
    # written as 4-byte numbers with empty 4 byte value before and afterwards
    # so the number of 4-byte values to read for each particle/ trajectory is 14+nspec
    nvals = 14 + nspec;

    # read all data as 4-byte int values
    particle_file = open(file, "rb")
    dummy = particle_file.read(3*4) # skip header
    data = particle_file.read(nvals*numpart*4)
    dataAsInt = struct.unpack('@'+nvals*numpart*'i', data)
    dataAsInt = np.reshape(dataAsInt, (numpart,nvals))
    particle_file.close()
    #read all data as 4-byte float values
    particle_file = open(file, "rb")
    dummy = particle_file.read(3*4) # skip header
    data = particle_file.read(nvals*numpart*4)
    dataAsFloat = struct.unpack('@'+nvals*numpart*'f', data)
    dataAsFloat = np.reshape(dataAsFloat,(numpart,nvals))
    particle_file.close()

    # select values for output
    npoint = dataAsInt[0:numpart,1]
    xyz = dataAsFloat[0:numpart,2:5]
    itramem = dataAsInt[0:numpart,5]
    vars = dataAsFloat[0:numpart,6:13]
    xmass = dataAsFloat[0:numpart,13:(13+nspec)]

    return [outheader, npoint, xyz, itramem, vars, xmass]

# --- FUNCTION sort_parcels
# It sorts the fields such that every element every element in the second axis of
# each array corresponds to one parcel. This is achieved using the identifier given
# in npoint, and taking into account how FLEXPART-WRF stores the trajectories
@jit(nopython=True)
def sort_parcels(npoint, fields):

    ntimes, numpart = npoint.shape
    nfields = len(fields)

    for t in range(1, ntimes, 1):
        npoint_t = npoint[t,:]-1
        numpart_t = np.argmax(npoint_t<0)
        if numpart_t==0:
            continue
        elif numpart_t<numpart:
            npoint_t_nneg = npoint_t[:numpart_t]
            for ifield in range(nfields):
                fields[ifield][t,npoint_t_nneg] = fields[ifield][t, :numpart_t]
                npoint_neg = np.delete(np.arange(numpart), npoint_t_nneg)
                fields[ifield][t,npoint_neg] = -1000.0

    return fields

#########################################################################################################

# Read release date of parcels from command line and define the path to FLEXPART output
release_date = sys.argv[1]
path = 'output/'+release_date+'/'
outpath = 'TRAJDATA/'

# Obtain a list of the files in path, and get the number of released parcels from the size
# of the first file named partposit_*
files = sorted(glob.glob(path+'partposit_*'))
files_sizes = []
for file in files:
    files_sizes.append(os.path.getsize(file))

numbytes = np.array(files_sizes)
numpart = np.round((numbytes/4+3)/15-1)
numpart = numpart.astype(int)[::-1]

# Get output times (dates and hours); by default release time is not included
with open(path + 'dates', 'r') as f:
    dates = f.readlines()

dates = [date.rstrip() for date in dates]
dates.insert(0, release_date+'0000')
ndates = len(dates)

# Create the HDF5 dataset in which trajectories will be stored
file = outpath+'traj'+release_date+'.h5', 'w'
f = h5py.File(file)
dset = f.create_dataset('trajdata', shape=(ndates, numpart[0], 9), dtype=np.float32)

# For all dates in dates files
# Reading particles positions from FLEXPART output file
print('2-Reading flexpart output...')
for t in range(ndates):
    numpart_t = numpart[t]
    out = readpart10(path, dates[t], numpart[t])
    dset[t,:numpart_t,0] = out[1]                 # parcel identifier
    dset[t,:numpart_t,1] = out[2][:,0]            # longitude
    dset[t,:numpart_t,2] = out[2][:,1]            # latitude
    dset[t,:numpart_t,3] = out[2][:,2]            # height
    dset[t,:numpart_t,4] = out[4][:,2]            # specific humidity
    dset[t,:numpart_t,5] = out[4][:,3]            # density
    dset[t,:numpart_t,6] = out[4][:,4]            # boundary layer height
    dset[t,:numpart_t,7] = out[4][:,6]            # temperature
    dset[t,:numpart_t,8] = out[5][:,0]            # mass
    print(t)

print('Number of dates : ', ndates)

# Next create a new file for the final datasets, save the parcel mass
newfile = file.replace('traj', 'newtraj')
g = h5py.File(newfile, 'w')
dset_mass = g.create_dataset('other_data', shape=(1,), maxshape=(None,))
dset_mass[0] = np.mean(f['trajdata'][:,:,8])

# In case of FLEXPART-WRF simulations, we need to sort the parcels
npoint = dset[:,:,0]               # parcel identifier
npoint = npoint.astype(int)
x = dset[:,:,1]                    # longitude in degrees
x[x>180.0] = x[x>180.0]-360.0
y = dset[:,:,2]                    # latitude in degrees
z = dset[:,:,3]                    # initial height in m
q = dset[:,:,4]                    # specific humidity in kg kg-1
rho = dset[:,:,5]                  # density in kg m-3
h_abl = dset[:,:,6]                # PBL height
T_k = dset[:,:,7]                  # temperature in K
f.close()

sorted_fields = sort_parcels(npoint, [x, y, q, z, h_abl, rho, T_k])
x = sorted_fields[0]; y = sorted_fields[1]; q = sorted_fields[2]
z = sorted_fields[3]; h_abl = sorted_fields[4]
rho = sorted_fields[5]; T_k = sorted_fields[6]

# Compute the relative humidity
R_d= 287.057                           # specific gas constant
R_w = 461.5                            # specific gas constant for water vapor
eps = R_d/R_w
w = q/(1.0-q)                          # mixing ratio
Tv = T_k/(1.0-w*(1.0-eps)/(w+eps))     # virtual temperature (from 3.16 and 3.59 Wallace)
p_pa = rho*R_d*Tv                      # from definition of virtual temperature
e = p_pa*w/(w+eps)                     # partial pressure of water vapor

# Compute the saturated vapor pressure using equation 10 from Bolton (T must be converted to ºC)
e_s = 611.2*np.exp(17.67*(T_k-273.15)/(T_k-273.15+243.5))
RH = 100*e/e_s
RH[RH<0] = -1000.0

# Final dataset
dset_final = g.create_dataset('trajdata', shape=(ndates, numpart[0], 6), dtype=np.float32)
dset_final[:,:,0] = x
dset_final[:,:,1] = y
dset_final[:,:,2] = z
dset_final[:,:,3] = h_abl
dset_final[:,:,4] = q
dset_final[:,:,5] = RH
g.close()
