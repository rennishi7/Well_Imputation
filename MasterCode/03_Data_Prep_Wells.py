# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:47:46 2020

@author: saulg
"""

import utils_well_data as wf  # file that holds the code for this project

New = True
Plot = True
root = "./Datasets/"
# Location must be added
# Options, as string: 'Escalante_Valley_Beryl_Enterprise_UT' 'Cedar_Valley_UT' 'Hueco_Bolson_UT' 'Yolo_Basin_CA'
Wells=wf.wellfunc()
# read the well data from a json file
wells_dict = Wells.read_well_pickle('Escalante_Valley_Beryl_Enterprise_UT')

# extractwelldata extracts waterlevel measurements and creates a panda data 
# Bcap and Fcap are bottom and final cap, this control guarrenties that wells
# will contain data before and after the caps
# MinEx is the minimum number examples required within dataset
# extract the data into a panda data frame
origwelldata_df, location_df, centroid = Wells.extractwelldata(wells_dict, Bcap=1948, Fcap=2019, MinExTotal=150)  
#well_names = origwelldata_df.columns         

# now need to resample well data to begining of month ('1MS') or chosen period
# next most used will be 'QS' Quarter Start Frequency
# need to fill with NaNs on large gaps
Well_Data = Wells.interp_well(origwelldata_df, gap_size = '365 days', pad = 90, spacing = '1MS')
# interpolate the well data to regular interval
# do not interpolate for large gaps (gap size can be set in function)
# provide data on either side of measured data - no nans - can be set in func
# can select data interval in function
wells_dict['Data'] = Well_Data

# check the wells
Wells.well_plot(origwelldata_df, Well_Data, plot_wells=Plot)  # plot the data to look at

#Save Datasets
if New == True:
    Wells.Save_Pickle(wells_dict, 'Well_Data')
    origwelldata_df.to_hdf(root + '03_Original_Points.h5', key='df', mode='w')



