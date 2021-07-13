# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:12:27 2020

@author: saulg
"""
import os
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from pandas import DatetimeIndex
from scipy import interpolate


class wellfunc():
    def __init__(self):
        if os.path.isdir('./Datasets') is False:
            os.makedirs('./Datasets')
        self.Data_root = "./Datasets/"
        
        if os.path.isdir('./Aquifers Data') is False:
            os.makedirs('./Aquifers Data')
            print('The Aquifer folder with data is empty')
        self.Aquifer_root = "./Aquifers Data/"
        
    '''###################################
                Well Data Sampling
    '''###################################

    ##### Opens well json and loads data
    def read_well_pickle(self, well_file):
        self.wellfile = self.Aquifer_root + well_file + '.pickle'
        with open(self.wellfile, 'rb') as handle:
            wells = pickle.load(handle)
        return wells
    
    def read_pickle(self, file, root):
        file = root + file + '.pickle'
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def Save_Pickle(self, Data, name:str):
        with open(self.Data_root + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def extractwelldata(self, wells,Bcap=1948,Fcap=2018, MinExTotal=50):
        if Bcap>Fcap:
            print('Error: Bottom Cap Year is greater than Final Cap year.')
            exit()
        #Needs to add error handle of caps boundaries outside of dataset size
        if MinExTotal<1:
            print('Error: Original Data dataframe must have at least 1 example.')
            exit()
        #Unpack Lat and Long, and Centroid
        location_df = wells['Location']
        centroid = wells['Centroid']
        
        #Create Emtpy Dataframe for Well Time Series
        well_fit = pd.DataFrame()
        #Create Caps as pandas time stamps to compare dates
        begtime = pd.Timestamp(dt.datetime(Bcap, 1, 1))# data before this date
        endtime = pd.Timestamp(dt.datetime(Fcap, 1, 1))# data after this date
        
        #Create Subset to speed algorithm
        #greater than the start date and smaller than the end date
        mask = (wells['Data'].index > begtime) & (wells['Data'].index < endtime)
        well_subset_test = wells['Data'].loc[mask]
        # Drop any column in subset that has less than MinExTotal non empty cells
        well_subset_test = well_subset_test.drop(well_subset_test.columns[well_subset_test.apply(lambda col: col.notnull().sum() < MinExTotal)], axis=1)
        #Creating Well subset Data
        well_fit = wells['Data'][well_subset_test.columns]
        #Create new Locations Dataframe
        location_df = location_df.loc[well_fit.columns]
        #Create new Centroid Data Frame
        centroid.loc['Latitude'][0] = location_df['Latitude'].min() + ((location_df['Latitude'].max() - location_df['Latitude'].min())/2)
        centroid.loc['Longitude'][0] = location_df['Longitude'].min() + ((location_df['Longitude'].max() - location_df['Longitude'].min())/2)
        self.CentroidY = round(centroid.loc['Latitude'][0], 1)
        self.CentroidX = round(centroid.loc['Longitude'][0], 1)
        return well_fit, location_df, centroid
        
    def interp_well(self, wells_df, gap_size = '365 days', pad = 90, spacing = '1MS'):
        # gaps bigger than this are set to nan
        # padding on either side of a measured point not set to nan, even if in a gap
        # spacing to interpolate to 1st day of the month (MS or month start)
        well_interp_df = pd.DataFrame()
        # create a time index to interpolate over - cover entire range
        interp_index: DatetimeIndex = pd.date_range(start=min(wells_df.index), freq=spacing, end=max(wells_df.index))
        # loop over each well, interpolate data using pchip
        for well in wells_df:
            temp_df = wells_df[well].dropna()  # available data for a well drops NaN
            x_index = temp_df.index.astype('int')  # dates for available data
            x_diff = temp_df.index.to_series().diff()  # data gap sizes
            fit2 = interpolate.pchip(x_index, temp_df)  # pchip fit to data
            ynew = fit2(interp_index.astype('int'))  # interpolated data on full range
            interp_df = pd.DataFrame(ynew, index=interp_index, columns=[well])
    
            # replace data in gaps of > 1 year with nans
            gaps = np.where(x_diff > gap_size)  # list of indexes where gaps are large
            
            # Can Probably optimize and remove this code
            # if padding plus start is larger than the gap size currently 1 year
            # Non nans will be replaced
            for g in gaps[0]:
                start = x_diff.index[g - 1] + dt.timedelta(days=pad)
                end = x_diff.index[g] - dt.timedelta(days=pad)
                interp_df[start:end] = np.nan
            beg_meas_date = x_diff.index[0]      # date of 1st measured point
            end_meas_date = temp_df.index[-1]    # date of last measured point
            mask1 = (interp_df.index < beg_meas_date)  # Check values before measured date
            interp_df[mask1] = np.nan  # blank out data before 1st measured point
            mask2 = (interp_df.index >= end_meas_date) # data from last measured point to end
            interp_df[mask2] = np.nan # blank data from last measured point
            
            # add the interp_df data to the full data frame
            well_interp_df = pd.concat([well_interp_df, interp_df], join="outer", axis=1, sort=False)
            
        # return a pd data frame with interpolated wells - gaps with nans
        return well_interp_df

    '''###################################
                Plotting Functions
    '''###################################
    def well_plot(self, combined_df, well_df, plot_wells):
        # plot some of the wells just to look
        well_names = well_df.columns
        if plot_wells:
            somewells = well_names[0:5] # first 5 wells
            well = well_names[-1]  # last well in the file
    
            combined_df.plot(y=somewells, style='.')
            well_df.plot(y=somewells, style='-')
            plt.title('Some example wells')
    
            plt.figure(3)
            plt.plot(combined_df[well], '*')
            plt.plot(well_df[well])
            plt.title('Measured and Interpolated data for well ' + str(well))
    
            plt.figure(4)
            plt.plot(well_df)
            plt.plot(combined_df[well_names], '-.')
            plt.title('Measured and Interpolated data for all wells and Satellite Data')

    
    def plot_sat_data(self, feature_df, plot_data):

        if plot_data:
            x=len(feature_df.columns)
            nrows = int(np.ceil(x/2))
            if x == 1:
                ncol = 1
            else: ncol = 2

        if len(feature_df.columns) > 1:
            fig, ax= plt.subplots(nrows,ncol)
            #ax=ax.flatten()
            for i,p in enumerate(feature_df.columns):
                df0=feature_df[[feature_df.columns[i]]]
                df0.plot(title=p,ax=ax[i])
                stats=feature_df[feature_df.columns[i]]
                print(feature_df.columns[i],' ',stats)
        else:
            plt.figure(5)
            plt.plot(feature_df.index, feature_df.iloc[:,0], title = feature_df.columns)
    
    '''###################################
                Satellite Resampling
    '''###################################
    
    def sat_resample(self, feature_df):
        # resamples the data from both datasets to a monthly value,
        # uses the mean of all measurements in a month
        # first resample to daily values, then take the start of the month
        feature_df = feature_df.resample('D').mean()
        # D is daily, mean averages any values in a given day, if no data in that day, gives NaN
    
        feature_df.interpolate(method='pchip', inplace=True, limit_area='inside')
        
        # MS means "month start" or to the start of the month, this is the interpolated value
        feature_df = feature_df.resample('MS').first()
        return feature_df


    '''###################################
                DATA PROCESSING METHODS
    '''###################################

    def DropNA(self, df):
        df.dropna(axis=0, inplace=True)
        return df
        
    def Rolling_Window(self, Feature_Data, Names, years=[1, 3, 5, 10]):
        # This loop adds the yearly, 3-year, 5-year, and 10-year rolling averages of each variable to the dataframe
        # rolling averages uses the preceding time to compute the last value, e.g., using the preceding 5 years of data to get todays
        # Causes the loss of X number of data points
        for name in Names:
            for year in years:
                new = name + '_rw_yr_' + str(year).zfill(2)
                Feature_Data[new] = Feature_Data[name].rolling(year * 12).mean()
        return Feature_Data.dropna()


    def sat_offsets(self, Feature_Data, Names, offsets=[0.5,1,2,3]):
        # This loop adds straight offsets of original features in yearly increments
        for name in Names:
            for year in offsets:
                new = name + '_offset_' + str(year).zfill(3)
                Feature_Data[new] = Feature_Data[name].shift(int(year * 12))
        return Feature_Data


    def sat_cumsum(self, Feature_Data, Names):
        for name in Names:
            new = name + '_int'
            Feature_Data[new] = (Feature_Data[name] - Feature_Data[name].mean()).cumsum()          
        return Feature_Data
    
    
    def Delete_Data(self, Feature_Data, DeleteNames=[]):
        # Deletes Features by name if desired
        # Must be input as list of strings where the strings are the name of the feautre
        if len(DeleteNames) > 0:
            for i in range(len(DeleteNames)):
                del Feature_Data[str(DeleteNames[i])]          
        return Feature_Data


