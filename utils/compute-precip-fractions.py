#!/mnt/netapp2/Store_uni/home/usc/fp/aco/.conda/envs/ENV/bin/python
# coding: utf-8

# Import modules
import sys
import numpy as np
import xarray as xr
import pandas as pd

# --- FUNCTION precip_fractions
# It takes a xarray dataset with the moisture source field and returns a list of precipitation fractions.
# It needs a set of predefined source regions in source_path, where there is a directory for each region
# with a netCDF file named trmask_d01_rg.nc in the same grid as the moisture source field.
# This is the usual setup when working with WRF-WVTs.
def precip_fractions(ms_ds):
    
    # Read results for moisture sources
    # In ms_2d we have the amount of liters evaporated per grid cell, so we do not need to weight by the area
    ms = ms_ds.ms_2d.values[0,:,:].copy()
    ms[0,0] = ms_ds.ms_3d_south.values
    ms[-1,0] = ms_ds.ms_3d_north.values
    dx = 0.25; nx = int(360/dx)
    dy = 0.25; ny = int(180/dy)+1

    # Define the number of sources and the path of the mask files
    nsources = 11
    source_path = '/home/usc/fp/aco/gfnlmeteo/JOBS/SIM_AR198710_WVTS_ET/RUNS/Sim-common/SOURCES/'

    # Names of the different sources
    propnames = ['Tropical land', 'Eurasia', 'North America', 'Tropical Pacific', 'Tropical Atlantic', 'Tropical Indic',\
                 'North Atlantic', 'North Pacific', 'Internal seas', '3D South', '3D North']

    # Now open the mask file for each source and compute the proportions
    # The last two masks are 3D, we need to treat them separately    
    prop = [0]*nsources
    for source in range(1,nsources+1,1):
        ds = xr.open_dataset(source_path+str(source)+'/trmask_d01_rg.nc', engine='netcdf4')
        
        if source==(nsources-1):
            trmask_rg = ds.TRMASK3D.sel(Times=0).values[0,:,:]
            prop[source-1] = 100.0*(ms*trmask_rg).sum()/ms.sum()
            ds.close()
        elif source==nsources:
            trmask_rg = ds.TRMASK3D.sel(Times=0).values[0,:,:]
            prop[source-1] = 100.0*(ms*trmask_rg).sum()/ms.sum()
            ds.close()
        else:
            trmask_rg = ds.TRMASK.sel(Times=0).values
            prop[source-1] = 100.0*(ms*trmask_rg).sum()/ms.sum()
            ds.close()

    return propnames, prop


if __name__=='__main__':
    filename = sys.argv[1]
    ms_ds = xr.open_dataset(filename, engine='netcdf4')
    propnames, prop = precip_fractions(ms_ds)
    data = {'Region': propnames, 'Fractional contribution': prop}
    df = pd.DataFrame(data)
    df.to_csv(filename.replace('.nc', '.csv'), sep=';', decimal=',')
