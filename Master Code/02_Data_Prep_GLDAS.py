# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:17:37 2021

@author: saulg
"""

import utils_satellite_data as usd
''''# GLADS
north_east_corner_lat = 89.875
south_west_corner_lat = -59.875
north_east_corner_lon = 179.875
south_west_corner_lon = -179.875
dx = 0.25
dy = 0.25

#PDSI
north_east_corner_lat = 77.5
south_west_corner_lat = -60.0
north_east_corner_lon = 180.0
south_west_corner_lon = -180.0
dx = 2.5
dy = 2.5'''


utils = usd.utils_netCDF(89.875, -59.875, 179.875, -179.875, 0.25, 0.25)
grid = utils.netCDF_Grid_Creation()
bounds = utils.Shape_Boundary('./Aquifer Shapes/Escalante_Beryl.shp')
cell_names = utils.Find_intercepting_cells(grid)
mask = utils.Cell_Mask(cell_names, grid)
dates = utils.Date_Index_Creation('1948-01-01')

parse = usd.grids_netCDF(File_String=False, Variable_String=False)
data_path = './Satellite Data Prep/subset_GLDAS_NOAH025_M_2.0_20210628_013227.txt'
variables = './Satellite Data Prep/variables_list.txt'
data_root = r'C:\Users\saulg\Desktop\Remote_Data\GLDAS'
Data = parse.Parse_Data(mask, dates, data_root=data_root, name_text=data_path, variable_path=variables)
utils.Save_Pickle(Data,'GLDAS_Data_Not_Validataed', './Datasets')
#Data = utils.read_pickle('GLDAS_Data_Not_Validataed', "./Datasets/")
Data = parse.Validate_Data(mask, Data)
utils.Save_Pickle(Data,'GLDAS_Data', './Datasets')


