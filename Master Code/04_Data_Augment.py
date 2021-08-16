# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:47:46 2020

@author: saulg
"""

import utils_data_augmentation # file that holds the code for this project

# This script will calculate modifications to datasets. Particularly this
# loadas the GLDAS Data pickle file, and performs a forward rolling window.
# This will cause the data to lose months of data equivilating to the largest
# window size. 

# Importing well object class
DA = utils_data_augmentation.Data_Augmentation()

# Load GLDAS Data
Data = DA.read_pickle('GLDAS_Data', root = './Datasets/')
cell_names = list(Data.keys())
cell_names.remove('Location') # Remove location

# Loop through every cell in the dataset performing augmentations. We only used
# FRWA, however, offsets and cumulative averages are available.
for i, cell in enumerate(cell_names):
    data_temp = Data[cell] 
    data_temp = DA.Rolling_Window(data_temp, data_temp.columns, years=[1, 3, 5])
    Data[cell] = data_temp

# Save Data
DA.Save_Pickle(Data, 'GLDAS_Data_Augmented')

