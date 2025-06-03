Scripts to compute the precipitation sources field for a single time using the SOD08 (based on Sodemann et al., 2008) and DB99 (based on Dirmeyer and Brubaker, 1999) diagnostic tools. 
Trajectories from FLEXPART or FLEXPART-WRF are supposed to be stored in a HDF5 file with a dataset "trajdata" of size (ntimes, numpart, 6), where:
- [:,:,0] is the longitude
- [:,:,1] is the latitude
- [:,:,2] is the height
- [:,:,3] is the boundary layer height
- [:,:,4] is the specific humidity
- [:,:,5] is the relative humidity
Another dataset, "other_data", stores the parcel mass for the specific FLEXPART simulation. The path to these trajectories is given in the namelist. Additionally, evaporation,
precipitable water and precipitation are supposed to be in another directory, path_data. The corresponding files are readed using the function interpol_tcw_evap_complete, the name
of the files and variables in them may need to be adjusted.

To reproduce the results of the manuscript, the moisture source field needs to be calculated for each time step (1 hour, 3 hours or 6 hours) in the precipitation event, accumulating along the
time dimension and scaling the resulting field to the precipitation observed.
