# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:47:46 2020

@author: saulg
"""

#import io
import pandas as pd
import os
import wellfuncsSR as wf  # file that holds the code for this project

New = True
Plot = True
# Location must be added
# Options, as string: 'Escalante' 'Cedar Valley'
if os.path.isdir('./Datasets') is False:
    os.makedirs('./Datasets')
    print('Data is not in Datasets folder')
root = "./Datasets/"

# Importing well object class
Wells=wf.wellfunc()

#Load Data
Well_Data = pd.read_hdf(root + 'Well_Data.h5')
Feature_Data = pd.read_hdf(root + 'Features_Raw.h5')
# Manually Delete features by name if needed
Delete_Feature = []
Feature_Data = Wells.Delete_Data(Feature_Data, Delete_Feature)

'''
###### Temporary GLDAS MERGE
GLDAS = pd.read_hdf(root + 'GLDAS.h5')
Feature_Data = pd.concat([Feature_Data, GLDAS], axis = 1, join = "outer")
Feature_Data.to_hdf(root + 'Features_Raw.h5', key='df', mode='w')
#Feature_Data = Wells.DropNA(Feature_Data)
###### TO BE REMOVED IN FUTURE
'''


###### names is a list that is apened to include augmented datasets
###### Duplicate is made in order to avoid augmentation of augmentations
Original_Datasets = list(Feature_Data.columns)


# compute rolling window averages (new point at the end of the window)
# specify how many, and window size in "years"
Feature_Data = Wells.Rolling_Window(Feature_Data, Original_Datasets, years=[1, 3, 5, 10])
Feature_Data = Wells.sat_offsets(Feature_Data, Original_Datasets, offsets=[.25,.5,1,3,5,10])


# add cumulative sums of the drought indexes to the combined data
Feature_Data = Wells.sat_cumsum(Feature_Data, Original_Datasets)

# Manually Delete features by name if needed
Delete_Feature = []
Feature_Data = Wells.Delete_Data(Feature_Data, Delete_Feature)

if New == True:
    root = "./Datasets/"
    #Save Datasets
    Feature_Data.to_hdf(root + 'Features_Aug.h5', key='df', mode='w')

# Cleaning Feature Dataset by droping Nan
Feature_Data = Wells.DropNA(Feature_Data)

if New == True:
    root = "./Datasets/"
    #Save Datasets
    Feature_Data.to_hdf(root + 'Features.h5', key='df', mode='w')
