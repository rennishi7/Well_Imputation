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

New = True
Plot = True
root = "./Datasets/"
# Location must be added
# Options, as string: 'Escalante' 'Cedar Valley'
Wells=wf.wellfunc()
mylat,mylon=Wells.get_lat_long(Location='Escalante')
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
Well_Data = Wells.interp_well(origwelldata_df, gap_size = '365 days', pad = 90, spacing = '1MS')  
# interpolate the well data to regular interval
# do not interpolate for large gaps (gap size can be set in function)
# provide data on either side of measured data - no nans - can be set in func
# can select data interval in function


# check the wells - well names are hardcoded in the function
Wells.well_plot(origwelldata_df, Well_Data, plot_wells=Plot)  # plot the data to look at


# set to false after running the first time to not hit the servers
# if set to False, it will read data from hd5 file created the 1st run
'''
Feature_Data = Wells.get_sat_data(new=New, PDSI=True, SoilMoist=True, 
                                  GLDAS = ['Tair_f_inst', 'Evap_tavg', 'ESoil_tavg', 'Rainf_f_tavg'],
                                  GLDAS_Server = 'BYU')
'''
Feature_Data = Wells.get_sat_data(new=New, PDSI=True, SoilMoist=True, 
                                  GLDAS = [],
                                  GLDAS_Server = 'BYU')

# plot the external input data
Wells.plot_sat_data(Feature_Data, plot_data=Plot)


# resample satellite data to monthly - 1st of the month (hard coded in function)
Feature_Data = Wells.sat_resample(Feature_Data)


if New == True:

    #Save Datasets
    Feature_Data.to_hdf(root + 'Features_Raw.h5', key='df', mode='w')
    Well_Data.to_hdf(root + 'Well_Data.h5', key='df', mode='w')
