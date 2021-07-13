# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:47:46 2020

@author: saulg
"""

import os
import utils_well_data as wf  # file that holds the code for this project

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
Data = Wells.read_pickle('GLDAS_Data', root)
cell_names = list(Data.keys())
cell_names.remove('Location')

for i, cell in enumerate(cell_names):
    data_temp = Data[cell]
    data_temp = Wells.Rolling_Window(data_temp, data_temp.columns, years=[1, 3, 5])
    Data[cell] = data_temp

Wells.Save_Pickle(Data, 'GLDAS_Data_Augmented')

