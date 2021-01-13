# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:47:46 2020

@author: saulg
"""

#import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wellfuncsSR as wf  # file that holds the code for this project

New = False

# Location must be added
# Options, as string: 'Escalante' 'Cedar Valley'
Wells=wf.wellfunc(Location='Escalante')
mylat,mylon=Wells.get_lat_long()
# read the well data from a json file
wells_raw = Wells.read_well_json()


# extractwelldata extracts waterlevel measurements and creates a panda data 
# Bcap and Fcap are bottom and final cap, this control guarrenties that wells
# will contain data before and after the caps
# MinEx is the minimum number examples required within dataset
# extract the data into a panda data frame
origwelldata_df = Wells.extractwelldata(wells_raw, Bcap=1975, Fcap=2005, MinEx=50)  
well_names = origwelldata_df.columns         

# now need to resample well data to begining of month ('1MS') or chosen period
# next most used will be 'QS' Quarter Start Frequency
# need to fill with NaNs on large gaps
well_df = Wells.interp_well(origwelldata_df, gap_size = '365 days', pad = 90, spacing = '1MS')  
# interpolate the well data to regular interval
# do not interpolate for large gaps (gap size can be set in function)
# provide data on either side of measured data - no nans - can be set in func
# can select data interval in function


# check the wells - well names are hardcoded in the function
Wells.well_plot(origwelldata_df, well_df, plot_wells=False)  # plot the data to look at


# set to false after running the first time to not hit the servers
# if set to False, it will read data from hd5 file created the 1st run
gldas_df, names =Wells.get_sat_data(new=False, PDSI=True, SoilMoist=True, Temp=False, Precip=False)


# plot the external input data
#Data Plotter not working again....
Wells.plot_sat_data(gldas_df, plot_data=False)


# resample satellite data to monthly - 1st of the month (hard coded in function)
gldas_df = Wells.sat_resample(gldas_df,write=True)


# compute rolling window averages (new point at the end of the window)
# specify how many, and window size in "years"
gldas_df, names = Wells.Rolling_Window(gldas_df, names, Spacing=[1, 3, 5, 10])
gldas_df, names = Wells.sat_offsets(gldas_df, names, offsets=[.25,.5,.75,1,1.5,2,3,5,10])
# add cumulative sums of the drought indexes to the combined data
gldas_df, names = Wells.sat_cumsum(gldas_df, names)

DeleteNames=[]
gldas_df, names = Wells.Delete_Data(gldas_df, names, DeleteNames)

# combine the  data from the wells and the satellite observations  to a single dataframe (combined_df)
# this will have a row for every measurement (on the start of the month) a column for each well,
# and a column for pdsi and soilw and their rolling averages, and potentially offsets
combined_df=Wells.Combine_Data_Frames(well_df,gldas_df,names,DropNA=True)
# drop rows where there are no satellite data
# pdsi stops in 2014, soilw starts in 1948
names = pd.DataFrame(names)


if New == True:
    root = "./Datasets/"
    
    #Data Split into Well information and Feature Information
    Well_Data = combined_df
    Feature_Data = combined_df
    Feature_Index = Feature_Data.index
    feature_names = names
    
    #Feature Mask Labels
    well_end1 = well_names[0]
    well_end2 = well_names[-1]
    
    #Data Splits
    Well_Data = Well_Data.loc[:,well_end1:well_end2]
    Feature_Data = Feature_Data.drop(Feature_Data.loc[:,well_end1:well_end2].columns,axis = 1)
    Year = Feature_Data['year']
    Feature_Data = Feature_Data.drop(['year'], axis = 1)
    
    #Save Datasets
    Feature_Data.to_hdf(root + 'Features.h5', key='df', mode='w')
    Well_Data.to_hdf(root + 'Well_Data.h5', key='df', mode='w')
