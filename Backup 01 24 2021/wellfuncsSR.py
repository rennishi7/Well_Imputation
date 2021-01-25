# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:12:27 2020

@author: saulg
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DatetimeIndex
from scipy import interpolate
import datetime as dt
from dateutil.relativedelta import relativedelta
import json
import urllib
from urllib import request
from xml.etree import cElementTree as ET
import copy
import os
import grids
import time

class wellfunc():
    def __init__(self):
        if os.path.isdir('./Datasets') is False:
            os.makedirs('./Datasets')
        self.Data_root = "./Datasets/"
        
        if os.path.isdir('./Aquifers') is False:
            os.makedirs('./Aquifers')
            print('The Aquifer folder with data is empty')
        self.Aquifer_root = "./Aquifers/"
        
        
    '''###################################
                Well Data Sampling
    '''###################################
        
    def get_lat_long(self, Location):
        
        ###### This process needs to be done automatically
        ###### Needs to be done with centroid of basin polygon
        self.Location = Location
        if self.Location=="Escalante":
            ############## BERYL-ENTERPRISE  #############
            self.wellfile = self.Aquifer_root + 'Escalante_Valley-Beryl-Enterprise.json'
            self.latitude = 37.6  # Beryl-Enterprise latitude
            self.longitude = -113.7  # Beryl-Enterprise longitude
        elif self.Location=="Cedar Valley":
            ############## CEDAR VALLEY  ##############
            self.wellfile = self.Aquifer_root + 'Cedar_Valley.json'
            self.latitude = 37.7  # Cedar Valley latitude
            self.longitude = -113.7  # Cedar Valleylongitude
        else:
            print("Error: The location you have selected does not exist.")
            exit()
        return self.latitude, self.longitude
        
    
    ##### Opens well json and loads data
    def read_well_json(self):
        wellfile=self.wellfile
        with open(wellfile, 'r') as f:
            wells = json.load(f)
        return wells

    
    def extractwelldata(self, wells,Bcap=1975,Fcap=2005,MinEx=50):
        # wells should be the result of reading a json file of the well data with water levels and time stamps
        # This code includes a for loop to add the timeseries data for each well in wells
        # aquifer which has data from before 1975 and after 2005 and more than 50 data samples
        # these timeseries data are returned as a pandas dataframe
        
        if Bcap>Fcap:
            print('Error: Bottom Cap Year is greater than Final Cap year.')
            exit()
        #Needs to add error handle of caps boundaries outside of dataset size
        if MinEx<1:
            print('Error: Original Data dataframe must have at least 1 example.')
            exit()
        combined_df = pd.DataFrame()
        begtime = dt.datetime(Bcap, 1, 1)  # data before this date
        endtime = dt.datetime(Fcap, 1, 1)  # data after this date
        for well in wells['features']:
            if 'TsTime' in well and len(well['TsTime']) > MinEx:  # only records with 50 measurements
                welltimes = pd.to_datetime(well['TsTime'], unit='s', origin='unix')  # pandas hangles negative time stamps
                if welltimes.min() < begtime and welltimes.max() > endtime:  # data longer than 1975 - 2005
                    name = str(well['properties']['HydroID'])
                    elev = well['properties']['LandElev']
                    wells_df = elev + pd.DataFrame(index=welltimes, data=well['TsValue'], columns=[name])
                    wells_df = wells_df[np.logical_not(wells_df.index.duplicated())]
                    try:
                        combined_df = pd.concat([combined_df, wells_df], join="outer", axis=1, sort=False)
                        combined_df.drop_duplicates(inplace=True)
                    except Exception as e:
                        print(e)
                        break
        return combined_df
        
    def interp_well(self, combined_df, gap_size = '365 days', pad = 90, spacing = '1MS'):
        # gaps bigger than this are set to nan
        # padding on either side of a measured point not set to nan, even if in a gap
        # spacing to interpolate to 1st day of the month (MS or month start)
        well_interp_df = pd.DataFrame()
        # create a time index to interpolate over - cover entire range
        interp_index: DatetimeIndex = pd.date_range(start=min(combined_df.index), freq=spacing, end=max(combined_df.index))
        # loop over each well, interpolate data using pchip
        for well in combined_df:
            temp_df = combined_df[well].dropna()  # available data for a well drops NaN
            x_index = temp_df.index.astype('int')  # dates for available data
            x_diff = temp_df.index.to_series().diff()  # data gap sizes
            fit2 = interpolate.pchip(x_index, temp_df)  # pchip fit to data
            ynew = fit2(interp_index.astype('int'))  # interpolated data on full range
            interp_df = pd.DataFrame(ynew, index=interp_index, columns=[well])
    
            # replace data in gaps of > 1 year with nans
            gaps = np.where(x_diff > gap_size)  # list of indexes where gaps are large
            
            # Can Probably optimize and remove this code
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
                Satellite Data Download
    '''###################################
        
    def getTimeBounds(self, url):
        # This function returns the first and last available time
        # from a url of a getcapabilities page located on a Thredds Server
        f = urllib.request.urlopen(url)
        tree = ET.parse(f)
        root = tree.getroot()
        # These lines of code find the time dimension information for the netcdf on the Thredds server
        dim = root.findall('.//{http://www.opengis.net/wms}Dimension')
        dim = dim[0].text
        times = dim.split(',')
        times.pop(0)
        timemin = times[0]
        timemax = times[-1]
        # timemin and timemax are the first and last available times on the specified url
        return timemin, timemax

    def getThreddsValue(self, server, layer, lat, lon):
        # This function returns a pandas dataframe of the timeseries values of a specific layer
        # at a specific latitude and longitude from a file on a Thredds server
        # server: the url of the netcdf desired netcdf file on the Thredds server to read
        # layer: the name of the layer to extract timeseries information from for the netcdf file
        # lat: the latitude of the point at which to extract the timeseries
        # lon: the longitude of the point at which to extract the timeseries
        # returns df: a pandas dataframe of the timeseries at lat and lon for the layer in the server netcdf file
        # calls the getTimeBounds function to get the first and last available times for the netcdf file on the server
        timemin, timemax = self.getTimeBounds(server + "?service=WMS&version=1.3.0&request=GetCapabilities")
        # These lines properly format a url request for the timeseries of a speific layer from a netcdf on
        # a Thredds server
        server = server + "?service=WMS&version=1.3.0&request=GetFeatureInfo&CRS=CRS:84&QUERY_LAYERS=" + layer
        server = server + "&X=0&Y=0&I=0&J=0&BBOX=" + str(lon) + ',' + str(lat) + ',' + str(lon + .001) + ',' + str(
            lat + .001)
        server = server + "&WIDTH=1&Height=1&INFO_FORMAT=text/xml"
        server = server + '&TIME=' + timemin + '/' + timemax
        f = request.urlopen(server)
        tree = ET.parse(f)
        root = tree.getroot()
        features = root.findall('FeatureInfo')
        times = []
        values = []
        for child in features:
            time = dt.datetime.strptime(child[0].text, "%Y-%m-%dT%H:%M:%S.%fZ")
            times.append(time)
            values.append(child[1].text)
    
        df = pd.DataFrame(index=times, columns=[layer], data=values)
        df[layer] = df[layer].replace('none', np.nan).astype(float)
        return df
    
    
    def datetime_to_float(self, d):
        # function to convert a datetime object to milliseconds since epoch
        epoch = dt.datetime.utcfromtimestamp(0)
        total_seconds = (d - epoch).total_seconds()
        return total_seconds
    
    
    def datetime_now(self):
        # function to convert a datetime object to milliseconds since epoch
        now = dt.datetime.now()
        past = now + relativedelta(months=-2)
        year = past.year
        month = past.month
        return year, month
    
    
    def get_sat_data(self, new=False, PDSI=False, SoilMoist=False, GLDAS=[], GLDAS_Server = 'BYU'):


        if new:  # Only hit the web serves if it is the first time - otherwise use local files
            # calls a function to return a data frame of the timeseries of the pdsi and soilw
            mylat=self.latitude
            mylon=self.longitude
            feature_df = pd.DataFrame()
            
            ###### These are NOAA Servers
            ###### DAI Palmer Draught Severity Hosted by NCAR
            if PDSI:
                server1 = "https://www.esrl.noaa.gov/psd/thredds/wms/Datasets/dai_pdsi/pdsi.mon.mean.selfcalibrated.nc"
                layer1 = "pdsi"  # name of data column to be returned
                df = self.getThreddsValue(server1, layer1, mylat, mylon)  # pdsi values
                feature_df = pd.concat([feature_df, df], join="outer", axis=1)
            
            ###### Soil Moisture provided by NOAA Climate Prediction Center
            if SoilMoist:
                server2 = "https://www.esrl.noaa.gov/psd/thredds/wms/Datasets/cpcsoil/soilw.mon.mean.nc"
                layer2 = "soilw"  # name of data column to be returned
                df = self.getThreddsValue(server2, layer2, mylat, mylon)  # soilw values
                feature_df = pd.concat([feature_df, df], join="outer", axis=1)
            
            # NASA GLDAS Determining if user has ased for 
            if len(GLDAS) > 0:                
                # you need to provide a list of data webstrings from which to extract the time series
                gldas_url_list = []
                #Potential GLDAS List of variables
                #GLDAS = ['Tair_f_inst', 'Evap_tavg', 'ESoil_tavg', 'Rainf_f_tavg']                
                
                Data_Year, Data_Month = self.datetime_now()
        
                # Determine Limits of Years
                # 1948 to Present - ~ 3 months
                for year in range(1948, Data_Year):
                  # Range of Month
                  if year < Data_Year:
                      Month_Limit = 13
                  else:
                      Month_Limit = Data_Month
                      for month in range(1, Month_Limit):
    
                        # BYU Server Instructions
                        if GLDAS_Server == 'BYU':
                            # GLDAS 2.1 string format
                            if year > 1999:
                                gldas_url_list.append(f'http://tethys-staging.byu.edu/thredds/dodsC/data/gldas/raw/GLDAS_NOAH025_M.A{year}{month:02}.021.nc4')
                            else:
                            #GLDAS 2.0 string format
                                gldas_url_list.append(f'http://tethys-staging.byu.edu/thredds/dodsC/data/gldas/raw/GLDAS_NOAH025_M.A{year}{month:02}.020.nc4')
                        elif GLDAS_Server =='NASA':
                            # GLDAS 2.1 string format
                            if year > 1999:
                                gldas_url_list.append(f'https://hydro1.gesdisc.eosdis.nasa.gov/opendap/GLDAS/GLDAS_NOAH025_M.2.1/{year}/GLDAS_NOAH025_M.A{year}{month:02}.021.nc4')
                            # GLDAS 2.0 string format
                            else:
                                gldas_url_list.append(f'https://hydro1.gesdisc.eosdis.nasa.gov/opendap/GLDAS/GLDAS_NOAH025_M.2.0/{year}/GLDAS_NOAH025_M.A{year}{month:02}.020.nc4')
                        
                    
            # the order of the dimensions in the file - you need to know this in advance
            dim_order = ('time', 'lat', 'lon', )
            df = pd.DataFrame()
            
            
            # create an instance of the TimeSeries class - it collects and stores metadata you need in order to extract time series subsets
            for item in range(len(GLDAS)):
                series = grids.TimeSeries(files=gldas_url_list, var=GLDAS[item], dim_order=dim_order)
            
            
                # if you are getting data from a remote source that requires authentication/log-in, you need to also provide a username/password
                # this is the user that i created to retrieve data for the gldas tethys app
                if GLDAS_Server == 'NASA':
                    series.user = 'tethysgldas'
                    series.pswd = 'KKP4E2sjTfQGsMX'
                
                
                # get a time series at a point by specifying the coordinates on each dimension (None means no selected coordinate/specific time step -> use all)
                df = series.point(None, self.latitude, self.longitude)
                df.index = df ['datetime']
                df = df.drop(['datetime'], axis = 1)
                df = df.rename(columns={"values": GLDAS[item]})
            feature_df = pd.concat([feature_df, df], join="outer", axis=1)
            
        else:
            print('read hdf5 file')
            #Loading Feature Dataset if run is not new
            feature_df = pd.read_hdf(self.Data_root + 'Features_Raw.h5.h5', 'df')


        print(feature_df.describe())
        return feature_df


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
            plt.title('Measured and Interpolated data for well ' + well)
    
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
            newnames = copy.deepcopy(Names)  # names is a list of the varibiles in the data frame, need to unlink for loop
            # This loop adds the yearly, 3-year, 5-year, and 10-year rolling averages of each variable to the dataframe
            # rolling averages uses the preceding time to compute the last value, e.g., using the preceding 5 years of data to get todays
            for name in Names:
                for year in years:
                    new = name + '_rw_yr_' + str(year).zfill(2)
                    Feature_Data[new] = Feature_Data[name].rolling(year * 12).mean()
                    newnames.append(new)
            return Feature_Data


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


