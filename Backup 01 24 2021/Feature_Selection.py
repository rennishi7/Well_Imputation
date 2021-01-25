# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:50:06 2020

@author: saulg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import ELM
from collections import Counter
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


#Timer
t0 = time.time()

#Data Import
New = False
plot = True
save_plot = True
if os.path.isdir('./Datasets') is False:
    os.makedirs('./Datasets')
    print('Data is not in Datasets folder')
Data_root = "./Datasets/"
Feature_Data = pd.read_hdf(Data_root + 'EEMDfeatures.h5')
Well_Data = pd.read_hdf(Data_root + 'Well_Data.h5')
Feature_Index = Feature_Data.index
well_names = Well_Data.columns


if New:
    #Backward Feature Selection
    #Inscert Loop Create 
    BestScore = np.empty((Well_Data.shape[1], 1))
    BestFeatures = pd.DataFrame(index = well_names, columns = Feature_Data.columns)
    BestFeatures = BestFeatures.fillna(-9999)
    BestFeatures_String = BestFeatures.fillna(-9999)
    
    for i in range(Well_Data.shape[1]):
    #for i in range(2):
        well_name_temp = str(well_names[i])
        Well1 = pd.DataFrame(Well_Data[well_name_temp], index = Feature_Index[:])
        Well1 = Well1.join(Feature_Data, how='outer')
        Well1_train = Well1.dropna(subset=[well_name_temp])
        Well1_train = (Well1_train-Well1_train.mean())/(Well1_train.max()-Well1_train.min())
        
        y = Well1_train[well_name_temp]
        X = Well1_train.drop([well_name_temp], axis = 1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        clf = ELM.ELM()
        
        '''
        clf = MLPRegressor(hidden_layer_sizes=(40), random_state=42,\
        max_iter=100,\
        shuffle=True,\
        activation='relu',\
        learning_rate='adaptive',\
        learning_rate_init=0.01,\
        validation_fraction=0.2,\
        momentum=0.9,\
        early_stopping=True, \
        n_iter_no_change=10,\
        solver='adam',\
        nesterovs_momentum=True)
        '''
        
        sfs = SFS(clf,\
            #k_features = (Feature_Data.shape[1]-3, Feature_Data.shape[1]), #Can be set to 1
            k_features = (3, Feature_Data.shape[1]), #Can be set to 1
            forward = False,
            floating = False,
            verbose = 0,
            scoring = 'neg_mean_absolute_error',
            cv = 10,
            n_jobs = -1).fit(X_train, y_train)
        score = sfs.k_score_
        best_features = sfs.k_feature_names_
        best_features_index = sfs.k_feature_idx_
        
        #Instance Plotting
        if plot:
            fig1 = plot_sfs(sfs.get_metric_dict(),kind='std_dev')
            plt.title(well_name_temp)
            plt.show()
            plt.close()
            #Plotting Save
            if save_plot:
                #Create Histogram
                if os.path.isdir('./Histogram Figures') is False:
                    os.makedirs('./Histogram Figures')
                fig_namepng = "./Histogram Figures/" + well_name_temp + '.png'
                plt.savefig(fig_namepng, format="png", dpi=600 )
        
        #Update Values
        #Score Log
        BestScore[i,0] = score
        
        #Feature Log
        bft = np.asarray(best_features_index)
        BestFeatures.iloc[i,0:len(bft)] = bft
        bfts = np.asarray(best_features)
        BestFeatures_String.iloc[i,0:len(bft)] = bft
    
    #Saving Files
    BestScore = pd.DataFrame(BestScore,index=Well_Data.columns)
    BestScore.to_hdf(Data_root + 'BestScoreFeatureReduction.h5', key='df', mode='w')
    BestFeatures.to_hdf(Data_root + 'BestFeatures.h5', key='df', mode='w')
else:
    BestFeatures = pd.read_hdf(Data_root + 'BestFeatures.h5')

#Create Histogram
if os.path.isdir('./Histogram Figures') is False:
    os.makedirs('./Histogram Figures')
histo_flat = BestFeatures.to_numpy().flatten()
histo_flat = np.delete(histo_flat, np.where(histo_flat == -9999))
Map = np.empty(shape=(Feature_Data.shape[1],2), dtype = 'object')
Map[:,0] = np.array([*range(0, Feature_Data.shape[1])])
Map[:,1] = Feature_Data.columns
letter_counts = Counter(histo_flat)
Histo_Data = pd.DataFrame.from_dict(letter_counts, columns=['Counts'], orient='index')
index_old = Histo_Data.index.to_numpy()
index_new = index_old.astype(dtype = object)
for i in range(Feature_Data.shape[1]):
    index_new = np.where(index_new == Map[i,0] , Map[i,1], index_new)
Histo_Data.index= np.ndarray.tolist(index_new)
Histo_Data = Histo_Data.sort_values('Counts', ascending = False)
plt.figure(figsize=(20, 18))
Histo_Data.plot(kind='bar', align ='edge',width = 0.75, legend = None)
plt.yticks(range(0, Histo_Data['Counts'].max()+1))
plt.xlabel('Feature')
plt.ylabel('Frequency')
plt.title('Most Prevolent Features')
fig_namepng = "./Histogram Figures/" + 'Histogram' + '.png'
plt.tight_layout()
plt.savefig(fig_namepng, format="png", dpi=600 )

t1 = time.time()
print('Run-time was: ' + str((t1-t0)/60) + ' minutes')

Key_Features = Histo_Data.index[Histo_Data['Counts'] > 0.75*len(BestFeatures)]
ReducedFeatures = Feature_Data.loc[:, Key_Features[:]]
ReducedFeatures.to_hdf(Data_root + 'ReducedFeatures.h5', key='df', mode='w')

#Beta, takes really long time.
'''

    clf = MLPRegressor(hidden_layer_sizes=(40), random_state=42,\
        max_iter=100,\
        shuffle=True,\
        activation='relu',\
        learning_rate='adaptive',\
        learning_rate_init=0.01,\
        validation_fraction=0.2,\
        momentum=0.9,\
        early_stopping=True, \
        n_iter_no_change=10,\
        solver='adam',\
        nesterovs_momentum=True)




#Exhaustive Feature Selection
efs = EFS(MLPRegressor(hidden_layer_sizes=(40),max_iter=1000,shuffle=False,\
    activation='relu',\
    learning_rate='constant',learning_rate_init=0.01,\
    validation_fraction=0.2,\
    momentum=0.9,\
    early_stopping=True, n_iter_no_change=100,\
    solver='adam',nesterovs_momentum=True,),\
    #EFS Hyperparameters
    min_features = 1,\
    max_features = Well1_train.shape[1]-1,\
    scoring = 'accuracy',\
    cv = None,\
    n_jobs=-1).fit(X_train, y_train)
print(efs.best_score_)
best_features = efs.best_feature_names_
best_features_index = efs.best_idx

plot_sfs(efs.get_metric_dict(),kind='std_dev')
'''