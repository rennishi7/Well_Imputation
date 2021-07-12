# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 07:01:54 2021

@author: saulg
"""

import os

months_left_path = r'C:\Users\saulg\Desktop\GLDAS\subset_GLDAS_NOAH025_M_2.0_20210628_013227.txt'
months_list = open(months_left_path).readlines()

GLDAS_path = r'C:\Users\saulg\Desktop\New folder'
GLDAS = os.listdir(GLDAS_path)

file_names = []
for f in GLDAS:
    file_names.append(f)
    
months_list = [i.replace('\n','') for i in months_list]

for i, month in enumerate(months_list):
    month = month[::-1]
    month = month.split('/')[0]
    month = month[::-1]
    if month in file_names: months_list[i] = []
months_list = list(filter(None, months_list))

textfile = open('months_left.txt','w')
for element in months_list:
    textfile.write(element + "\n")
textfile.close()
