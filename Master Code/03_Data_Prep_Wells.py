# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:47:46 2020

@author: saulg
"""

import utils_well_data as wf  # file that holds the code for this project

# This script opens preprocessed pickle file containing 3 DataFrames: Centroid,
# Well Timeseries, and well location.
# High level does 3 things:
# 1: Extracts Well Time series, and drops any well without a minimum amount
#    of Data within a user specified window
# 2: Interpolates subset of data, removing gaps larger than user specified size
#    Gaps will be filled in using a machine learning algorithm. This step
#    is based on the assumption that groundwater changes are slow, and readings
#    will remain valid for a time period after the original reading.
# 3: Contains support functions of data augmentation such as Rolling-Window 
#    averages and plotting. 


# Data Locations
data_root = './Datasets/'
aquifer_root = './Aquifers Data'
figures_root = './Figures Aquifer'

# Location must be added
Wells=wf.wellfunc(data_root, aquifer_root, figures_root)

# read the well data from a pickle file
# Options, as string: 'Escalante_Valley_Beryl_Enterprise_UT' 'Cedar_Valley_UT' 
# 'Hueco_Bolson_UT' 'Yolo_Basin_CA'
raw_wells_dict = Wells.read_well_pickle('Escalante_Valley_Beryl_Enterprise_UT')

# extractwelldata extracts waterlevel measurements and creates a panda data 
# Bcap and Fcap are bottom and final cap, this control guarrenties that wells
# will contain data before and after the caps
# MinEx is the minimum number examples required within dataset
# extract the data into a panda data frame
wells_dict = Wells.extractwelldata(raw_wells_dict, Left=1948, Right=2019, Min_Obs_Months=150)          

# now need to resample well data to begining of month ('1MS') or chosen period
# next most used will be 'QS' Quarter Start Frequency
# need to fill with NaNs on large gaps
# interpolate the well data to regular interval
# do not interpolate for large gaps (gap size can be set in function)
# provide data on either side of measured data - no nans - can be set in func
# can select data interval in function
wells_dict['Data'] = Wells.interp_well(wells_dict['Data'], gap_size = '365 days', pad = 90, spacing = '1MS')


# Plot Well Results
Wells.well_plot(wells_dict['Data'], wells_dict['Data'], plot_wells= True)  # plot the data to look at

#Save Datasets
Wells.Save_Pickle(wells_dict, 'Well_Data')
wells_dict['Data'].to_hdf(data_root + '03_Original_Points.h5', key='df', mode='w')