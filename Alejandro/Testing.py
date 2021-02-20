import sys
import pandas as pd
from sklearn import svm # pylint: disable=import-error
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier # pylint: disable=import-error
from sklearn.ensemble import RandomForestClassifier # pylint: disable=import-error
from sklearn.model_selection import train_test_split # pylint: disable=import-error
from sklearn.neural_network import MLPClassifier # pylint: disable=import-error
from sklearn.preprocessing import StandardScaler # pylint: disable=import-error
from sklearn.naive_bayes import GaussianNB # pylint: disable=import-error
from sklearn.metrics import accuracy_score # pylint: disable=import-error
from sklearn.utils import shuffle # pylint: disable=import-error
import numpy as np
import os
import math
import winsound


for i in range(5):
    for n_back in range(4):
        if n_back == 1:
            continue
        else:
            pd.read_csv(f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\EEG_Data\\ERP_VP00{i+1}_{n_back}back.csv', header=None).T.to_csv(f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\EEG_Data\\ERP_VP00{i+1}_{n_back}-back.csv', header=False, index=False)
       
# call_get_data = False
# while not call_get_data:
#     dataIS = input('Enter type of Data (NIRS or EEG): ')
#     if dataIS == 'NIRS' or dataIS == 'EEG':
#         call_get_data = True
        
# extract_path = f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\{dataIS}_Data'
# read_path = f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\{dataIS}_Data\\All_Data\\All_Data.csv'


# try:
#     f = open(read_path)
#     all_data_file_exists = True
#     f.close()
# except FileNotFoundError:
#     print("Could not open")