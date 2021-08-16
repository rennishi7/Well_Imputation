# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:50:49 2021

@author: saulg
"""

import numpy as np
import os
import pickle
import gstools as gs  # "conda install -c conda-forge gstools"
import matplotlib
import matplotlib.pyplot as plt
import netCDF4
matplotlib.use('Qt5Agg')

class krigging_interpolation():
    # Establish where files are located. If location doesn't exist, it will 
    # create a new location.
    def __init__(self, data_root ='./Datasets', figures_root = './Figures Spatial'):
        # Data Root is the location where data will loaded from. Saved to class
        # in order to reference later in other functions.
        if os.path.isdir(data_root) is False:
            os.makedirs(data_root)
        self.data_root = data_root
        
        # Fquifer Root is the location to save figures.
        if os.path.isdir(figures_root) is False:
            os.makedirs(figures_root)
        self.figures_root = figures_root

    # Opens generic pickle file based on file path and loads data.
    def read_pickle(self, file, root):
        file = root + file + '.pickle'
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    # Saves generic pickle file based on file path and loads data.
    def Save_Pickle(self, Data, name:str):
        with open(self.Data_root + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=4)

    def create_grid_coords(self, x_c, y_c, num_cells, lag_multiple = 5):
        # create grid coordinates for kriging, make x and y steps the same
        # x_steps is the number of cells in the x-direction
        # Currently the number of x_steps is set. However, you could set the grid size
        # or make this a selectble attribute
        cell_size = np.absolute((max(x_c) - min(x_c))/num_cells)  # determine step size (positive)
        lag_distance = lag_multiple*cell_size
        grid_x = np.arange(min(x_c)-lag_distance, max(x_c)+lag_distance, cell_size)
        grid_y = np.arange(min(y_c)-lag_distance, max(y_c)+lag_distance, cell_size)
        return grid_x, grid_y


    def netcdf_setup(self, x_coords, y_coords, timestamp, filename):
        # setup a netcdf file to store the time series of rasters
        # copied from other lab code - you probably don't need everything here
        
        file = netCDF4.Dataset(self.data_root + '/' + filename, 'w', format="NETCDF4")
        
        lon_len  = len(x_coords)  # size of grid in x_dir
        lat_len  = len(y_coords)  # size of grid in y_dir
        
        time = file.createDimension("time", None) # time dimension - can extend e.g., size=0
        lat  = file.createDimension("lat", lat_len)  # create lat dimension in netcdf file of len lat_len
        lon  = file.createDimension("lon", lon_len)  # create lon dimension in netcdf file of len lon_len
        
        time      = file.createVariable("time", np.float64, ("time"))
        latitude  = file.createVariable("lat", np.float64, ("lat"))  # create latitude varilable
        longitude = file.createVariable("lon", np.float64, ("lon")) # create longitude varilbe
        tsvalue   = file.createVariable("tsvalue", np.float64, ('time', 'lon', 'lat'), fill_value=-9999)
        tsvalue.units = 'feet'
        
        latitude[:] = y_coords[:] 
        longitude[:] = x_coords[:]

    
        latitude.long_name = "Latitude"
        latitude.units = "degrees_north"
        latitude.axis = "Y"
        
        longitude.long_name = "Longitude"
        longitude.units = "degrees_east"
        longitude.axis = "X"
        
        timestamp = list(timestamp.to_pydatetime())
        units = 'days since 0001-01-01 00:00:00'
        calendar = 'standard'
        time[:] = netCDF4.date2num(dates = timestamp, units = units, calendar= calendar)
        time.axis = "T"
        time.units = units
        
        return file, tsvalue
    
    
    def fit_model_var(self, coords_df, x_c, y_c, values):
        # Check this: https://www.youtube.com/watch?v=bRj3HnEa1Z4&ab_channel=GeostatsGuyLectures
        # fit the model varigrom to the experimental variogram
        # this will fit a model variogram to experiemental data
        # however, occasionaly , there isn't a good fit.
        # the current version specifies a vargiogram rather than fitting one
        # the other code is still here, just commented out.
        # For the hard-coded variogram, we assume the range is 1/8 the size of the
        # area we are interested in
    
        bin_num = 20  # number of bins in the experimental variogram
        # first get the coords and determine distances
        x_delta = max(x_c) - min(x_c)  # distance across x coords
        y_delta = max(y_c) - min(y_c)  # distance across y coords
        max_dist = np.sqrt(np.square(x_delta + y_delta)) / 2  # assume correlated over 1/4 of distance
    
        # setup bins for the variogram
        bins_c = np.linspace(0, max_dist, bin_num)  # bin edges in variogram, bin_num of bins
    
        # compute the experimental variogram
        bin_cent_c, gamma_c = gs.vario_estimate_unstructured((x_c, y_c), values, bins_c)
        # bin_center_c is the "lag" of the bin, gamma_c is the value
    
        # # if fitting the variogram, this gets rid of 0 and low values
        # gamma_c = smooth(gamma_c, 5) # smooth the experimental variogram to help fitting - moving window with 5 samples
    
        # #----------------------------------------
        # # fit the model variogram
        # # potential models are: Gaussian, Exponential, Matern, Rational, Stable, Linear,
        # #                       Circular, Spherical, and Intersection
        # fit_var = gs.Linear(dim=1)
    
        # fit_var.fit_variogram(bin_cent_c, gamma_c, nugget=False)  # fit the model variogram
    
    
        # # check to see of range of varigram is very small (problem with fit), it so, set it to a value
        # # also check to see if it is too long and set it to a value
        # if fit_var.len_scale < max_dist/3:  # check if too short
        #     fit_var.len_scale = max_dist/3  # set to value
        #     #print('too short, set len_scale to: ', max_dist/3)
        # elif fit_var.len_scale > max_dist*1.5: # check if too long
        #     fit_var.len_scale = max_dist       # set to value
        #     #print('too long, set len_scale to: ', max_dist)
        # # End of variogram fitting stuff
    
        data_var = np.var(values)
        data_std = np.std(values)
        fit_var = gs.Stable(dim=2, var=data_var, len_scale=max_dist/4, nugget=data_std)
        # this line creates a variogram that has a range of 1/8 the longest distance
        # across the data; above we divided the max_dist by 2, here we divide by 4 so
        # 1/8
        # the code commented out above "fits" the variogram to the actual data
        # here we just specifiy the range, with the sill equal to the
        # variation of the data, and the nugget equal to the standard deviation.
        # the nugget is probably too big using this approach
        # we could se the nugget to 0
    

        # plot the variogram to show fit and print out variogram paramters
        # should make it save to pdf file
        ax1 = fit_var.plot(x_max=max_dist)  # plot model variogram
        ax1.plot(bin_cent_c, gamma_c)  # plot experimental variogram
        plt.show()
        print(fit_var)  # print out model variogram parameters.
        return fit_var
    
    def krig_field_generate(self, var_fitted, x_c, y_c, values, grid_x, grid_y):
        # use GSTools to krig  the well data, need coords and value for each well
        # use model variogram paramters generated by GSTools
        # fast - is faster the variogram fitting
        krig_map = gs.krige.Ordinary(var_fitted, cond_pos=[x_c, y_c], cond_val=values)
        krig_map.structured([grid_x, grid_y]) # krig_map.field is the numpy array of values
        return krig_map
    
