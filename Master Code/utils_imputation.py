# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:23:13 2021

@author: saulg
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
#import HydroErr as he  # BYU hydroerr package - conda installable
import os



class utils_imputation():
    def __init__(self, Fig_root = "./Satellite Figures/"):
        if os.path.isdir(Fig_root) is False:
            os.makedirs(Fig_root)
        self.Fig_root = Fig_root 
        return
    
    def read_pickle(self, pickle_file, pickle_root):
        wellfile = pickle_root + pickle_file + '.pickle'
        with open(wellfile, 'rb') as handle:
            wells = pickle.load(handle)
        return wells
    
    def Save_Pickle(self, Data, name:str, path:str):
        with open(path + '/' + name + '.pickle', 'wb') as handle:
            pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def Data_Split(self, Data, well_name_temp, Shuffle=False):
        if Shuffle:
             Data = Data.sample(frac=1) #The frac keyword argument specifies the fraction of rows to return in the random sample
        Y = Data[well_name_temp].to_frame()
        X = Data.drop(well_name_temp, axis=1)
        return Y, X
    
    def Data_Join(self, pd1, pd2, method='outer', axis=1):
        return pd.concat([pd1, pd2], join='outer', axis=1)
    
    def metrics(self, Metrics, n_wells):
        metrics_result = Metrics.sum(axis = 0)
        normalized_metrics = metrics_result/n_wells
        with open(self.Fig_root + 'Error_Aquifer.txt', "w") as outfile:
            print('Raw Model Metrics:  \n' + str(metrics_result), file=outfile)
            print('\n Normalized Model Metrics Per Well:  \n' + str(normalized_metrics), file=outfile)
        np.savetxt(self.Fig_root + 'Error_Wells.txt', Metrics.values, fmt='%d')
        return metrics_result, normalized_metrics

    def Model_Training_Metrics_plot(self, Data, name):
        pd.DataFrame(Data).plot(figsize=(8,5))
        plt.grid(True)
        plt.gca().set_ylim(-0.5, 5)
        plt.savefig(self.Fig_root + name + '_Training_History')
        plt.show()
    
    def Q_Q_plot(self, Prediction, Observation, name): #y_test_hat, y_test):
        #Plotting Prediction Correlation
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        plt.scatter(Prediction, Observation)
        plt.ylabel('Observation')
        plt.xlabel('Prediction')
        plt.legend(['Prediction', 'Observation'])
        plt.title('Prediction Correlation: ' + name)
        limit_low = 0
        limit_high = 1
        cor_line_x = np.linspace(limit_low, limit_high, 9)
        cor_line_y = cor_line_x
        plt.xlim(limit_low, limit_high)
        plt.ylim(limit_low, limit_high)
        plt.plot(cor_line_x, cor_line_y, color='r')
        ax1.set_aspect('equal', adjustable='box')
        plt.savefig(self.Fig_root + name + '_01_Q_Q')
        plt.show() 

    def observeation_vs_prediction_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "r")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='b')
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Prediction', 'Observation'])
        plt.title('Observation Vs Prediction: ' + name)
        plt.savefig(self.Fig_root + name + '_02_Observation')
        plt.show()
    
    def observeation_vs_imputation_plot(self, Prediction_X, Prediction_Y, Observation_X, Observation_Y, name):
        plt.figure(figsize=(12, 8))
        plt.plot(Prediction_X, Prediction_Y, "r")
        plt.plot(Observation_X, Observation_Y, label= 'Observations', color='b')
        plt.ylabel('Groundwater Surface Elevation')
        plt.xlabel('Date')
        plt.legend(['Prediction', 'Imputation'])
        plt.title('Observation Vs Imputation: ' + name)
        plt.savefig(self.Fig_root + name + '_03_Imputation')
        plt.show()

    def raw_observation_vs_prediction(self, Prediction, Raw, name, Aquifer):
        plt.figure(figsize=(6,2))
        plt.plot(Prediction.index, Prediction, 'b-', label='Prediction', linewidth=0.5)
        plt.scatter(Raw.index, Raw, label= 'Observations', color='salmon')
        plt.title(Aquifer + ': ' + 'Well: ' + name + ' Raw vs Model')
        plt.legend(fontsize = 'x-small')
        plt.tight_layout(True)
        plt.savefig(self.Fig_root + name + '_04_Prediction_vs_Raw')
        plt.show()
    
    def raw_observation_vs_imputation(self, Prediction, Raw, name):
        plt.figure(figsize=(6,2))
        plt.plot(Prediction.index, Prediction, 'b-', label='Prediction', linewidth=0.5)
        plt.scatter(Raw.index, Raw, label= 'Observations', color='salmon')
        plt.title('Well: ' + name + ' Raw vs Model')
        plt.legend(fontsize = 'x-small')
        plt.tight_layout(True)
        plt.savefig(self.Fig_root + name + '_05_Imputation_vs_Raw')
        plt.show()

    def Aquifer_Plot(self, imputed_df):
        plt.figure(figsize=(6,2))
        col_names = imputed_df.columns
        for i in range(len(imputed_df.columns)):
            plt.plot(imputed_df.index, imputed_df[col_names[i]], '-.')
        plt.title('Measured and Interpolated data for all wells')