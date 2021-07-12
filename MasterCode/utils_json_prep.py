# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:09:04 2021

@author: saulg
"""
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

def read_well_json(wellfile):
    with open(wellfile, 'r') as f:
        wells = json.load(f)
    return wells

def JSON_to_Pandas(wells):
    location_df = pd.DataFrame(columns=['Longitude','Latitude'])
    centroid = pd.DataFrame([0,0], index=(['Longitude','Latitude']))
    combined_df = pd.DataFrame()
    location_temp = pd.DataFrame([[0,0]], columns=['Longitude','Latitude'])
    loop = tqdm(total = len(wells['features']), position = 0, leave = False)
    for i, well in enumerate(wells['features']):
        try:
            if 'TsTime' in well and 'TsValue' in well and len(well['geometry']['coordinates'])==2:
                assert len(well['TsTime']) == len(well['TsValue'])
                welltimes = pd.to_datetime(well['TsTime'], unit='s', origin='unix')  # pandas hangles negative time stamps
                name = str(well['properties']['HydroID'])
                elev = well['properties']['LandElev']
                wells_df = elev + pd.DataFrame(index=welltimes, data=well['TsValue'], columns=[name])
                wells_df = wells_df[np.logical_not(wells_df.index.duplicated())]
                combined_df = pd.concat([combined_df, wells_df], join="outer", axis=1, sort=False)
                combined_df.drop_duplicates(inplace=True)
                location_temp.iloc[0,0] = well['geometry']['coordinates'][0]
                location_temp.iloc[0,1] = well['geometry']['coordinates'][1]
                location_temp.index = [name]
                location_df = pd.concat([location_df, location_temp]) 
        except Exception as e:
            print(e)
            pass
        centroid.loc['Longitude'] = location_df['Longitude'].min() + ((location_df['Longitude'].max() - location_df['Longitude'].min())/2)
        centroid.loc['Latitude'] = location_df['Latitude'].min() + ((location_df['Latitude'].max() - location_df['Latitude'].min())/2)
        loop.update(1)
    return combined_df, location_df, centroid

def Pandas_to_Pickle(combined_df, location_df, centroid, name):
    Aquifer = {'Data':combined_df, 'Location':location_df, 'Centroid':centroid}
    pickle_name = name + '.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(Aquifer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return Aquifer

    
name = 'Escalante_Valley_Beryl_Enterprise_UT.json'
wells_raw = read_well_json(name)
combined_df, location_df, centroid = JSON_to_Pandas(wells_raw)
Aquifer = Pandas_to_Pickle(combined_df, location_df, centroid, name = name[:-5])

