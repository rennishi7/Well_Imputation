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


utils = usd.utils_netCDF(77.5, -60.0, 180.0, -180.0, 2.5, 2.5)
grid = utils.netCDF_Grid_Creation()
bounds = utils.Shape_Boundary('./Aquifer Shapes/Gulf_Coast_TX.shp')
cell_names = utils.Find_intercepting_cells(grid)
mask = utils.Cell_Mask(cell_names, grid)
dates = utils.Date_Index_Creation('1850-01-01')

parse = usd.grids_netCDF(File_String=True, Variable_String=True)
data_path = r'C:\Users\saulg\Desktop\Remote_Data\sc_PDSI_pm\pdsisc.monthly.maps.1850-2018.fawc-1.r2.5x2.5.ipe-2.nc'
variables = 'sc_PDSI_pm'
Data = parse.Parse_Data(mask, dates, data_path = data_path, data_root=None, 
                        name_text=None, variable_name=variables, variable_path=None)
utils.Save_Pickle(Data,'PDSI_Data_Not_Validated', './Datasets')
#Data = utils.read_pickle('PDSI_Data_Not_Validated', "./Datasets/")
Data = parse.Validate_Data(mask, Data)
utils.Save_Pickle(Data,'PDSI_Data', './Datasets')


