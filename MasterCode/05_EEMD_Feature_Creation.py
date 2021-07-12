# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:06:16 2020

@author: saulg
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import utils_well_data as wf
from PyEMD import EEMD

New = True
Plot = True
# Location must be added
# Options, as string: 'Escalante' 'Cedar Valley'
if os.path.isdir('./Datasets') is False:
    os.makedirs('./Datasets')
    print('Data is not in Datasets folder')
root = "./Datasets/"

# Importing well object class
Wells=wf.wellfunc()

#Load Data
Data = Wells.read_pickle('PDSI_Data', root)
cell_names = list(Data.keys())
cell_names.remove('Location')

for i, cell in enumerate(cell_names):
    data_temp = Data[cell]
    for j, var in enumerate(data_temp.columns):
        if __name__ == "__main__":
            eemd = EEMD(trials= 10, noise_width = 0.05, ext_EMD=None, parallel = False, separate_trends=False)
            eIMF = eemd(data_temp[var].values).T

        
        #Convert pdsi EEMD to Pandas
        eIMF = pd.DataFrame(eIMF, index=data_temp.index)
        label = []
        for k in range(eIMF.shape[1]):
            lab = var + '_eemd_' + str(k+1)
            label.append(lab)
        eIMF = eIMF.set_axis(label,axis=1,inplace=False)
        data_temp = data_temp.join(eIMF, how='outer')
        
        
        #Plot EEMD Results
        fig, axs = plt.subplots(nrows= eIMF.shape[1]+1, ncols=1, figsize=(12,18))
        fig.suptitle(str('Ensemble Empirical Mode Decomposition: ' + cell +' ' + var))
        plt.subplots_adjust(top=.95, hspace=0.25)
        for k, _ in enumerate(data_temp):
            if k == 0:
                axs[k].plot(data_temp.index, data_temp[data_temp.columns[k]])
                axs[k].set(ylabel = var)
            else:
                axs[k].plot(data_temp.index, data_temp[data_temp.columns[k]])
                axs[k].set(ylabel = 'IMF ' + str(k-1))
        plt.show()
        
        
        #Saving Figure
        if Plot:
            if os.path.isdir('./EEMD Figures') is False:
                os.makedirs('./EEMD Figures')
            fig_namepng = "./EEMD Figures/" + str(cell) + '_EEMD' + '.png'
            fig.savefig(fig_namepng, format="png", dpi=600 )

        Data[cell] = data_temp
if New:
    Wells.Save_Pickle(Data, 'PDSI_Data_EEMD')



