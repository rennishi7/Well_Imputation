# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 00:24:06 2021

@author: saulg
"""

import pandas as pd
import ELM
import time
import numpy as np
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor


#Timer
t0 = time.time()


#Data Settings
New = False
plot = True
save_plot = True
Data_root = "./Datasets/"


###### Import Imputed Dataset
'''
Feature_Data_Wells = pd.read_hdf(Data_root + 'Imputed_data.h5')
###### Optimal features based on Backward search
Feature_Data_Features = pd.read_hdf(Data_root + 'ReducedFeatures.h5')
###### Combining Features and Well Imputes
Feature_Data_Complete = pd.concat([Feature_Data_Wells, Feature_Data_Features], join='inner', axis=1)
'''
Feature_Data = pd.read_hdf(Data_root + 'Features_Aug.h5')
Feature_Data.dropna(axis=0, inplace=True)
#Feature_Data_Complete = pd.read_hdf(Data_root + 'Imputed_data.h5')

###### Measured Well Data
#Well_Data = pd.read_hdf(Data_root + 'Well_Data.h5')
Well_Data = pd.read_hdf(Data_root + 'Features_Raw.h5')
Well_Data = Well_Data['pdsi'].to_frame()

###### Getting Well Dates
Feature_Index = Feature_Data.index
###### Getting Well Names for Looping
well_names = Well_Data.columns
###### Getting Feature Names
Feature_Names = Feature_Data.columns


###### Model Setup
elm = ELM.ELM()


###### Importing Metrics and Creating Error DataFrame
Sum_Metrics = pd.DataFrame(columns=['ME','MAPE','RMSE','MSE','r2'])
###### Creating Empty Imputed DataFrame
Imputed_Data = pd.DataFrame(index=Feature_Index)
###### Creating Empty Imputed DataFrame
Gap_Data = pd.DataFrame(index=Feature_Index)

for i in range(Well_Data.shape[1]):
    Method = 'min_max' #min_max, or standard
    ###### Obtaining Well Name
    well_name_temp = str(well_names[i])
    ###### Get Well readings for single well
    Well_set_temp = pd.DataFrame(Well_Data[well_name_temp], index = Feature_Index[:])
    ###### Getting all Well Nans to fill and merge with Well Data
    Gap_Data = Well_set_temp[Well_set_temp[well_name_temp].isna()]
    ###### Joining Features to Well Data
    Well_set_temp = Well_set_temp.join(Feature_Data, how='outer')
    ###### Dropping all missing values for training
    Well_train = Well_set_temp.dropna(subset=[well_name_temp])

    ###### Dataset required to unnormalize dataset
    y_renorm = Well_train[well_name_temp].to_frame()
    ###### Normalizing Well Measurements
    Well_train = elm.Data_Normalization(Well_train, method = Method)


    ###### Creating Gap where to evaluate model, may be random or user input
    Cut_Date1, Cut_Date2 = elm.Define_Gap(Well_train, BegDate=[], gap=5, Random=True)
    ###### Modifying Result Dataframe to only inclue values within Gap    
    Evaluation_set = elm.cut_training_data(Well_train, Cut_Date1, Cut_Date2)
    y_truth = Evaluation_set[well_name_temp].to_frame()
    
    ###### Removing Test set from Training Set
    Well_train = pd.concat([Well_train, Evaluation_set]).drop_duplicates(keep=False)
    ###### Dataset shuffling and split between training features and training labels
    X1, y1 = elm.Shuffle_Split(Well_train, well_name_temp, Shuffle = True)
    ###### Dataset normalization option max min or z-score
    X2 = elm.Data_Normalization(Feature_Data, method = Method)
    
    
    ###### ELM Training and prediction   
    model = elm.fit(X1,y1)
    y2 = pd.DataFrame(elm.predict(X2), index=Feature_Index[:], columns=['Prediction'])
    '''
    ###### MLP Training and prediction    
    model = clf.fit(X1,y1)
    y2 = pd.DataFrame(clf.predict(X2), index=Feature_Index[:], columns=['Prediction'])
    #'''
    
    ###### Renormalizing Predictions
    y2 = elm.renorm_data(y2, y_renorm, method = Method)    
    ###### Renorming Measured Score Values
    y_truth = elm.renorm_data(y_truth, y_renorm, method = Method)
    ###### Creating Data set based on predictions this will be scored
    ###### Joining and triming datasets to same window as y_truth for score
    Result = pd.concat([y_truth, y2], axis =1, join='inner')

    
    ###### Scoring Function
    score = elm.score(Result[well_name_temp], Result['Prediction'])
    ###### Additional Metrics For single well
    well_metrics = elm.metrics(Result[well_name_temp], Result['Prediction'], well_name_temp)

    ###### Logging Metrics for all well
    Sum_Metrics = Sum_Metrics.append(well_metrics)
    ###### Creating Dataset of imputed Data
    ###### Generating original clean well dataset with gaps
    Well_with_gaps = pd.DataFrame(Well_Data[well_name_temp], index = Feature_Index[:])
    ###### Filling in gaps with imputed Data
    Filled_time_series = Well_with_gaps[well_name_temp].fillna(y2['Prediction'])
    Imputed_Data = pd.concat([Imputed_Data, Filled_time_series], join='inner', axis=1)
    
    ###### Plotting Single Well 
    if plot:
        Well_Figure = elm.plot_imputed_results(y2, y_renorm, [well_name_temp], Cut_Date1, Cut_Date2, Save_plot = save_plot)

###### Metrics and Dataset Saving
metrics_result = Sum_Metrics.sum(axis = 0)
print(metrics_result) 
