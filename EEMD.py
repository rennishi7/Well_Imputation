# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:06:16 2020

@author: saulg
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from PyEMD import EEMD

#Settings
New = True
save_plot = True
if os.path.isdir('./Datasets') is False:
    os.makedirs('./Datasets')
    print('Data is not in Datasets folder')
root = "./Datasets/"
DataSets_EEMD = ['pdsi','soilw']

#Load Data
Well_Data = pd.read_hdf(root + 'Well_Data.h5')
Feature_Data = pd.read_hdf(root + 'Features.h5')
Feature_Index = Feature_Data.index
Year = Feature_Data.index

#EEMD For Documentation and hyperparameters visit
#https://pyemd.readthedocs.io/en/latest/eemd.html
for i in range(len(DataSets_EEMD)):
    if __name__ == "__main__":
        eemd = EEMD(trials= 100, noise_width = 0.05, ext_EMD=None, parallel = False)
        dt1 = Feature_Data[DataSets_EEMD[i]].to_numpy()
        eIMF = eemd(dt1).T
    
    #Convert pdsi EEMD to Pandas
    eIMF = pd.DataFrame(eIMF, index=Feature_Index[:])
    label = []
    for j in range(eIMF.shape[1]):
        lab = 'e' + DataSets_EEMD[i] + '_' + str(j+1)
        label.append(lab)
    eIMF = eIMF.set_axis(label,axis=1,inplace=False)
    
    #Plot EEMD Results
    plt.figure(figsize=(12, 18))
    plt.subplots_adjust(top=.95, hspace=0.25)
    plt.subplot(eIMF.shape[1]+1, 1, 1)
    plt.plot(Year, Feature_Data[DataSets_EEMD[i]])
    plt.ylabel(DataSets_EEMD[i])
    if save_plot:
        if os.path.isdir('./EEMD Figures') is False:
            os.makedirs('./EEMD Figures')
        fig_namepng = "./EEMD Figures/" + str(DataSets_EEMD[i]) + '_EEMD' + '.png'
        plt.savefig(fig_namepng, format="png", dpi=600 )
    
    #Join Feature Dataset
    for j in range(eIMF.shape[1]):
        plt.subplot(eIMF.shape[1]+1, 1, j+2)
        df = eIMF.iloc[:,j]
        plt.plot(Year, df)
        plt.ylabel('IMF ' + str(j+1))
    plt.suptitle(DataSets_EEMD[i] + ' EEMD')
    
    Feature_Data = Feature_Data.join(eIMF, how='outer')

if New:
    Feature_Data.to_hdf(root + 'EEMDfeatures.h5', key='df', mode='w')



