# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 23:23:13 2021

@author: saulg
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import HydroErr as he  # BYU hydroerr package - conda installable
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
import os


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
            X2 = (Feature_Data-Feature_Data.min())/(Feature_Data.max()-Feature_Data.min())
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
        #assert in_df.shape[1] == ref_df.shape[1], 'must have same # of columns'
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
        beg_date = train_df.index[0]  # date to begin plot
        end_date= train_df.index[-1]  # date to end plot
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
            plt.plot(train_df[beg_date : date1],  'b+', ms=4, label='Training Data')
            plt.plot(train_df[(date2 + ' Dec') : end_date],  'b+', ms=4)
            
            
            # Training Data within the Gap
            plt.plot(train_df[date1 : date2], color='darkorange', 
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