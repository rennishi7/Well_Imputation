# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:32:26 2020

@author: saulg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wellfuncsSR as wf
import time
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from copy import deepcopy

#Timer
t0 = time.time()

#Data Import
New = True
plot = True
save_plot = True
Data_root = "./Datasets/"

#Feature_Data = pd.read_hdf(Data_root + 'Features.h5')
#Feature_Data = pd.read_hdf(Data_root + 'EEMDfeatures.h5')
Feature_Data = pd.read_hdf(Data_root + 'ReducedFeatures.h5')
Well_Data = pd.read_hdf(Data_root + 'Well_Data.h5')

Feature_Index = Feature_Data.index
well_names = Well_Data.columns
Feature_Names = Feature_Data.columns

#Model Setup
elm = wf.ELM()

clf = MLPRegressor(hidden_layer_sizes=(25), random_state=42,\
max_iter=100,\
shuffle=True,\
activation='relu',\
learning_rate='adaptive',\
learning_rate_init=0.01,\
momentum=0.9,\
early_stopping=True, \
n_iter_no_change=10,\
solver='adam',\
nesterovs_momentum=True)


Sum_Metrics = pd.DataFrame(columns=['ME','MAPE','RMSE','MSE','r2'])
Imputed_Data = pd.DataFrame(index=Feature_Index)
for i in range(Well_Data.shape[1]):
#for i in range(2):
    well_name_temp = str(well_names[i])
    Well1 = pd.DataFrame(Well_Data[well_name_temp], index = Feature_Index[:])
    Well1 = Well1.join(Feature_Data, how='outer')
    Well1_train = Well1.dropna(subset=[well_name_temp])
    y_renorm = Well1_train[well_name_temp].to_frame()
    Well1_train = (Well1_train-Well1_train.mean())/(Well1_train.max()-Well1_train.min())

    X1 = Well1_train.drop([well_name_temp], axis = 1).values
    y1 = Well1_train[well_name_temp].values
    X2 = (Feature_Data-Feature_Data.mean())/(Feature_Data.max()-Feature_Data.min())
    
    #model = elm.fit(X1,y1)
    #y2 = elm.predict(X2)
    
    model = clf.fit(X1,y1)
    y2 = clf.predict(X2)
    

    Test = pd.DataFrame(y2, index = X2.index, columns=['Test'])
    Test = elm.renorm_data(Test, y_renorm)
    y2 = deepcopy(Test)
    y2.columns = [well_name_temp]
    Train = pd.DataFrame(y1, index=Well1_train.index, columns=['Train'])
    Train = elm.renorm_data(Train, y_renorm)
    Result = pd.concat([Train,Test], join='inner', axis=1)
    
    Cut_Date1, Cut_Date2 = elm.Define_Gap(Test, BegDate=[], gap=5, Random=True)
    Result = elm.cut_training_data(Result, Cut_Date1, Cut_Date2)
    
    score = elm.score(Result['Train'], Result['Test'])
    well_metrics = elm.metrics(Result['Train'], Result['Test'],well_name_temp)

    Sum_Metrics = Sum_Metrics.append(well_metrics)
    Imputed_Data = pd.concat([Imputed_Data, y2], join='inner', axis=1)
    
    
    Well_Figure = elm.plot_imputed_results(y2, Train,  [well_name_temp], Cut_Date1, Cut_Date2)

Sum_Metrics.to_hdf(Data_root + 'Metrics.h5', key='metrics', mode='w')
metrics_result = Sum_Metrics.sum(axis = 0)
Imputed_Data.to_hdf(Data_root + 'Imputed_data.h5', key='imputed_df', mode='w')  # create the data.h5 file, overwrite if exists
elm.Aquifer_Plot(Imputed_Data)
print(metrics_result) 
