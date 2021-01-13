# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:12:27 2020

@author: saulg
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as plt_pdf
from pandas import DatetimeIndex
from scipy import interpolate
import datetime as dt
import json
import urllib
from urllib import request
from xml.etree import cElementTree as ET
import copy
import HydroErr as he  # BYU hydroerr package - conda installable
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from scipy.stats import zscore
import os

class wellfunc():
    def __init__(self,Location):
        self.Location=Location
        if os.path.isdir('./Datasets') is False:
            os.makedirs('./Datasets')
        self.Data_root = "./Datasets/"
        
        if os.path.isdir('./Aquifers') is False:
            os.makedirs('./Aquifers')
            print('The Aquifer folder with data is empty')
        self.Aquifer_root = "./Aquifers/"
        
    def get_lat_long(self):
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
        
    def read_well_json(self):
        wellfile=self.wellfile
        with open(wellfile, 'r') as f:
            wells = json.load(f)
        return wells
    
    def Combine_Data_Frames(self,well_df, gldas_df, Data_Names, DropNA=True, ):
        combined_df=pd.concat([well_df,gldas_df], join="outer", axis=1, sort=False)
        if DropNA:
            combined_df.dropna(subset=Data_Names, inplace=True)
        return combined_df
    
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
        self.well_names=combined_df.columns 
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
            beg_intrp_date = interp_df.index[0]  # date of 1st pchip interped data
            end_intrp_date = interp_df.index[-1] # date of last pchip interped data
            mask1 = (interp_df.index < beg_meas_date)  # Check values before measured date
            interp_df[mask1] = np.nan  # blank out data before 1st measured point
            mask2 = (interp_df.index >= end_meas_date) # data from last measured point to end
            interp_df[mask2] = np.nan # blank data from last measured point
            
            # add the interp_df data to the full data frame
            well_interp_df = pd.concat([well_interp_df, interp_df], join="outer", axis=1, sort=False)
            
        # return a pd data frame with interpolated wells - gaps with nans
        return well_interp_df

    def well_plot(self, combined_df, well_df, plot_wells):
        # plot some of the wells just to look
        well_names=self.well_names
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
        # total_seconds will be in decimals (millisecond precision)
        return total_seconds
    
    
    def get_sat_data(self, new=False, PDSI=False, SoilMoist=False, Temp=False, Precip=False):
        self.pdsi=PDSI
        self.SoilMoist=SoilMoist
        self.Temp=Temp
        self.Precip=Precip
        self.Inputs=[]
        if self.pdsi:
            self.Inputs.append('pdsi')
        if self.SoilMoist:
            self.Inputs.append('soilw')
        if self.Temp:
            self.Inputs.append('temp')
        if self.Precip:
            self.Inputs.append('precip')
        self.SatData=self.Inputs
        
        
        if new:  # Only hit the web serves if it is the first time - otherwise use local files
            # calls a function to return a data frame of the timeseries of the pdsi and soilw
            mylat=self.latitude
            mylon=self.longitude
            SatData = pd.DataFrame()
            gldas_df = pd.DataFrame()
            SatData.to_hdf(self.Data_root + 'SatData.h5', key='dF', mode='w')  # create the data.h5 file, overwrite if exists
            if PDSI:
                server1 = "https://www.esrl.noaa.gov/psd/thredds/wms/Datasets/dai_pdsi/pdsi.mon.mean.selfcalibrated.nc"
                layer1 = "pdsi"  # name of data column to be returned
                df = self.getThreddsValue(server1, layer1, mylat, mylon)  # pdsi values
                df.to_hdf(self.Data_root + 'SatData.h5', key='pdsi', mode='a')  # append data to the file
                gldas_df = pd.concat([gldas_df, df], join="outer", axis=1)
            if SoilMoist:
                server2 = "https://www.esrl.noaa.gov/psd/thredds/wms/Datasets/cpcsoil/soilw.mon.mean.nc"
                layer2 = "soilw"  # name of data column to be returned
                df = self.getThreddsValue(server2, layer2, mylat, mylon)  # soilw values
                df.to_hdf(self.Data_root + 'SatData.h5', key='soilw', mode='a')  # append data to the file
                gldas_df = pd.concat([gldas_df, df], join="outer", axis=1)
                #NOT WORKING
            if Temp:
                server2 = "https://www.esrl.noaa.gov/psd/thredds/wms/Datasets/cpcsoil/soilw.mon.mean.nc"
                layer2 = "soilw"  # name of data column to be returned
                df = self.getThreddsValue(server2, layer2, mylat, mylon)  # soilw values
                df.to_hdf(self.Data_root + 'SatData.h5', key='temp', mode='a')  # append data to the file
                gldas_df = pd.concat([gldas_df, df], join="outer", axis=1)
                #NOT WORKING
            if Precip:
                server2 = "https://www.esrl.noaa.gov/psd/thredds/wms/Datasets/cpcsoil/soilw.mon.mean.nc"
                layer2 = "soilw"  # name of data column to be returned
                df = self.getThreddsValue(server2, layer2, mylat, mylon)  # soilw values
                df.to_hdf(self.Data_root + 'SatData.h5', key='precip', mode='a')  # append data to the file
                gldas_df = pd.concat([gldas_df, df], join="outer", axis=1)
            print('got data from server')
        else:
            # joins the two dataframes (df and df1) together into the same timeseries dataframe
            print('read hdf5 file')
            gldas_df = pd.DataFrame()
            if PDSI:
                pdsi=pd.read_hdf(self.Data_root + 'SatData.h5', 'pdsi')
                gldas_df = pd.concat([gldas_df, pdsi], join="outer", axis=1)
            if SoilMoist:
                soilw=pd.read_hdf(self.Data_root + 'SatData.h5', 'soilw')                
                gldas_df = pd.concat([gldas_df, soilw], join="outer", axis=1)
            if Temp:
                temp=pd.read_hdf(self.Data_root + 'SatData.h5', 'temp')
                gldas_df = pd.concat([gldas_df, temp], join="outer", axis=1)
            if Precip:
                precip=pd.read_hdf(self.Data_root + 'SatData.h5', 'precip')
                gldas_df = pd.concat([gldas_df, precip], join="outer", axis=1)

        print(gldas_df.describe())
        names=list(gldas_df.columns)
        return gldas_df, names

    
    def plot_sat_data(self, gldas_df, plot_data):
        
        if plot_data:
            x=len(self.Inputs)
            nrows = int(np.ceil(x/2))
            if x == 1:
                ncol = 1
            else: ncol = 2

            fig, ax= plt.subplots(nrows,ncol)
            ax=ax.flatten()
            for i,p in enumerate(self.Inputs):
                df0=gldas_df[[self.Inputs[i]]]
                df0.plot(title=p,ax=ax[i])
                stats=gldas_df[self.Inputs[i]]
                print(self.Inputs[i],' ',stats)
    
    def sat_resample(self, gldas_df,write=True):
        # resamples the data from both datasets to a monthly value,
        # uses the mean of all measurements in a month
        # first resample to daily values, then take the start of the month
        gldas_df = gldas_df.resample('D').mean()
        # D is daily, mean averages any values in a given day, if no data in that day, gives NaN
    
        gldas_df.interpolate(method='pchip', inplace=True, limit_area='inside')
    
        gldas_df = gldas_df.resample('MS').first()
        # MS means "month start" or to the start of the month, this is the interpolated value
        if write:
            gldas_df.to_hdf(self.Data_root + 'SatData.h5', key='gldas_df', mode='a')  # append data to the file
        # print(gldas_df.describe())
        return gldas_df

                        
        '''###################################
                    DATA PROCESSING METHODS
        '''###################################
        
    def Rolling_Window(self, gldas_df, names, Spacing=[1, 3, 5, 10]):

            years=Spacing
            newnames = copy.deepcopy(names)  # names is a list of the varibiles in the data frame, need to unlink for loop
            # This loop adds the yearly, 3-year, 5-year, and 10-year rolling averages of each variable to the dataframe
            # rolling averages uses the preceding time to compute the last value, e.g., using the preceding 5 years of data to get todays
            for name in names:
                for year in years:
                    new = name + '_yr' + str(year).zfill(2)
                    gldas_df[new] = gldas_df[name].rolling(year * 12).mean()
                    newnames.append(new)
            return gldas_df, newnames


    def sat_offsets(self, gldas_df, DataNames, offsets=[.25,.5,.75,1,1.5,2,3]):
        names = DataNames
        Offset_Names = self.Inputs
        for name in Offset_Names:
            for year in offsets:
                new = name + '_offset' + str(year).zfill(3)
                gldas_df[new] = gldas_df[name].shift(int(year * 12))
                names.append(new)
        return gldas_df, names

    def sat_cumsum(self, gldas_df, names):
        for i in range(len(self.Inputs)):
            name=str(self.Inputs[i]+'_int')
            gldas_df[name] = (gldas_df[self.Inputs[i]] - gldas_df[self.Inputs[i]].mean()).cumsum()
            names.append(name)  # add new column name            
        gldas_df['year']=gldas_df.index.year + (gldas_df.index.month - 1)/ 12 # add linear trend (decimal years)
        names.append('year')
        return gldas_df, names
    
    def Delete_Data(self, gldas_df, names, DeleteNames=[]):
        if DeleteNames:
            for i in range(len(DeleteNames)):
                del gldas_df[str(DeleteNames[i])]
                names.remove(DeleteNames[i])            
        return gldas_df, names
    

        '''###################################
                DATA Normalization
        '''###################################
        
    def norm_method(self,Method):
        #Options are normalize, or zscore
        self.norm_train_method=Method
        return str(self.norm_train_method)
        
    def norm_training_data(self, in_df, ref_df):
        
        if self.norm_train_method=='normalize':
            norm_in_df = (in_df - ref_df.min().values) / (
                        ref_df.max().values - ref_df.min().values)  # use values as df sometimes goofs
        if self.norm_train_method=='zscore':
            norm_in_df=(in_df-in_df.mean)/in_df.std 
        return norm_in_df
        
    '''###################################
                Learning Methods
    '''###################################

class ELM(BaseEstimator, RegressorMixin):
    
    def __init__(self, hidden_units = 500, lamb_value = 100, Fig_root = "./Impute Figures/"):
        if os.path.isdir('./Impute Figures/') is False:
            os.makedirs('./Impute Figures/')
        self.Fig_root = Fig_root 
        self.hidden_units = hidden_units
        self.lamb_value = lamb_value
        return
    
    def fit(self, X, y):
         x = X
         x1 = np.column_stack(np.ones(x.shape[0])).T  # bias vector of 1's
         tx = np.hstack((x, x1))  # training matrix
         ty = y
         input_length = tx.shape[1]
         
         W_in, b, W_out = self._fit_matrix_multiplication(input_length, tx, ty, Try = 0)
         self.W_in = W_in 
         self.b = b
         self.W_out = W_out
         return
     
    def _fit_matrix_multiplication(self, input_length, tx, ty, Try = 0):
         W_in = np.random.normal(size=[input_length, self.hidden_units])
         b = np.random.normal(size=[self.hidden_units])

         # now do the matrix multiplication
         Xm = self.input_to_hidden_(tx, W_in, b)  # setup matrix for multiplication, it is a function
         I = np.identity(Xm.shape[1])
         I[Xm.shape[1] - 1, Xm.shape[1] - 1] = 0
         I[Xm.shape[1] - 2, Xm.shape[1] - 2] = 0
         #Recurrent Function so invalid weights cannot crash algorithm
         if Try < 10:
             try:
                 W_out = np.linalg.lstsq(Xm.T.dot(Xm) + self.lamb_value * I, Xm.T.dot(ty), rcond=-1)[0]
             except:
                 Try += 1
                 W_in, b, W_out = self._fit_matrix_multiplication(input_length, tx, ty, Try)
         else:
            print('Could Not Create ELM Matrix')
            pass
         return W_in, b, W_out
        
        
    def Shuffle_Split(self, Well1_train, well_name_temp, Shuffle = False):
         if Shuffle:
             Well1_train = Well1_train.sample(frac=1)
         X1 = Well1_train.drop([well_name_temp], axis = 1).values
         y1 = Well1_train[well_name_temp].values
         return X1, y1
     
    def Data_Normalization(self, Feature_Data, method = 'min_max'):
        if method == 'min_max':
            X2 = (Feature_Data-Feature_Data.mean())/(Feature_Data.max()-Feature_Data.min())
        elif method == 'standard':
            X2 = (Feature_Data-Feature_Data.mean())/(Feature_Data.std(ddof=0))
        return X2
        
    def input_to_hidden_(self, x, Win, b):
        # setup matrixes
        a = np.dot(x, Win) + b
        a = np.maximum(a, 0, a)  # relu
        return a
        
    def transform(self,):
        return
        
    def predict(self, X):
        x1 = np.column_stack(np.ones(X.shape[0])).T
        X = np.hstack((X, x1))
        x = self.input_to_hidden_(X, self.W_in, self.b)
        y = np.dot(x, self.W_out)
        return y
        
    def score(self, y1, y2):
        well_r2 = he.r_squared(y1, y2) #Coefficient of Determination
        score_r2 = well_r2
        return score_r2
    
    def metrics(self, y1, y2, well_name):
        metrics=pd.DataFrame(columns=['ME','MAPE','RMSE','MSE','r2'], index = [well_name])
        #Create Dataframe
        well_me = he.me(y1, y2) #Mean Error
        well_mape = he.mape(y1, y2) #Mean Absolute Percentage Error
        well_rmse = he.rmse(y1, y2) #Root Mean Squared Error
        well_mse = he.mse(y1, y2) #Mean Squared Error
        well_r2 = he.r_squared(y1, y2) #Coefficient of Determination
        
        #Create empty Dataframe
        metrics.iloc[0,0] = well_me
        metrics.iloc[0,1] = well_mape
        metrics.iloc[0,2] = well_rmse
        metrics.iloc[0,3] = well_mse
        metrics.iloc[0,4] = well_r2
        
        return metrics


    def cut_training_data(self, combined_df, date1, date2):
        train_df = combined_df[(combined_df.index > date1) & (combined_df.index < date2)]
        if len(train_df) == 0:
            if self.Random == True:
                date1, date2 = self.Define_Gap(combined_df, self.BegDate, self.gap, self.Random)
                self.Date1 = date1
                self.Date2 = date2
                train_df = self.cut_training_data(combined_df, self.Date1, self.Date2)
            else:
                print('At least one of the wells has no points in the specified range')
        return train_df
    
    
    def renorm_data(self, in_df, ref_df, method = 'min_max'):
        assert in_df.shape[1] == ref_df.shape[1], 'must have same # of columns'
        if method == 'min_max':
            renorm_df = (in_df * (ref_df.max().values - ref_df.min().values)) + ref_df.min().values
        elif method == 'standard':
            renorm_df = (in_df * (ref_df.std(ddof=0).values)) + ref_df.mean().values
        return renorm_df     
    
    
    def Define_Gap(self, combined_df, BegDate, gap=7, Random=True):
        self.BegDate = BegDate
        self.gap = gap
        self.Random = Random
        if Random and not BegDate:
            temp1=str(combined_df.index[0])  
            temp1=temp1[0:4]
            temp1=int(temp1)  
                          
            temp2=str(combined_df.index[len(combined_df.index)-1])
            temp2=temp2[0:4]
            temp2=int(temp2)
            
            #np.random.seed(42)
            GapStart=np.random.randint(temp1,(temp2-gap))
            date1=str(GapStart)              #random gap
            date2=str(GapStart+gap)      #random gap
        else:
            if BegDate:
                date1=str(BegDate[0])
                date2=str(int(BegDate[0])+gap)
            else:
                date1='1995'
                date2='2001'
        self.Date1 = date1
        self.Date2 = date2
        
        return date1, date2
    
    
    def plot_imputed_results(self, imputed_df, train_df, well_name_temp, date1, date2, Save_plot = True):
        if self.Random:
            date1 = self.Date1
            date2 = self.Date2
        else:
            date1=str(date1)
            date2=str(date2)
        beg_date='1960'  # date to begin plot
        end_date='2015'  # date to end plot
        date = date1 + ' Dec'  # date go through the end of the year
        fig_size = (6,2) # figure size in inches
        #im_plots = plt_pdf.PdfPages('Imputed_well_plots.pdf')

        with open(self.Fig_root + 'Error_data.txt', "w") as outfile:
            print("Begin date:  ", beg_date, ", End date:  ,", end_date, file=outfile)
            print("Well,                  ME,     MAPE,     RMSE,       R2", file=outfile)
            
        for well in well_name_temp:
            i_name = str(well_name_temp) + '_imputed'
            fig_namesvg = self.Fig_root + str(well_name_temp) + '.svg'
            fig_namepng = self.Fig_root + str(well_name_temp) + '.png'
            plt.figure(figsize=(6,2))
            
            # Full Imputation
            plt.plot(imputed_df, 'b-', label='Prediction', linewidth=0.5)
            
            # lines at begining and end of training data: Testing Gap
            plt.axvline(dt.datetime.strptime(date, '%Y %b'), linewidth=0.25)  # line at beginning of test period
            plt.axvline(dt.datetime.strptime((date2 + ' Dec'), '%Y %b'), linewidth=0.25)    # line at end of test period
            
            # training data: Original Points with Gaps
            plt.plot(train_df[beg_date : date],  'b+', ms=4, label='Training Data')
            plt.plot(train_df[(date2 + ' Dec') : end_date],  'b+', ms=4)
            
            
            # Training Data within the Gap
            plt.plot(train_df[date : date2], color='darkorange', 
                     marker=".", ms=3, linestyle='none', label='Target')
            
            # plot setup stuff
            plt.title('Well: ' + str(well_name_temp[0]))
            plt.legend(fontsize = 'x-small')
            plt.tight_layout(True)

            if Save_plot:
            # save plots
                plt.show()
                plt.close()
                plt.savefig(fig_namesvg, format="svg")
                plt.savefig(fig_namepng, format="png", dpi=600 )


        return
        
    def Aquifer_Plot(self, imputed_df):
        plt.figure(figsize=(6,2))
        col_names = imputed_df.columns
        for i in range(len(imputed_df.columns)):
            plt.plot(imputed_df.index, imputed_df[col_names[i]], '-.')
        plt.title('Measured and Interpolated data for all wells')
        return
