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
from window_slider import Slider
import numpy as np
import os
import math
import winsound
duration = 5000  # milliseconds
freq = 740  # Hz

# dataset = r'C:\Users\owner\Documents\GitHub\Capstone_490\Alejandro\TrainingData\Merged\Merged.csv'
# extract_path = r'C:\Users\owner\Documents\GitHub\Capstone_490\Alejandro\NIRS_Data'
# read_path = r'C:\Users\owner\Documents\GitHub\Capstone_490\Alejandro\NIRS_Data\All_Data\All_Data.csv'
rows_avg = 10
count = 1
hold = pd.DataFrame()
files = []
all_data_file_exists = False
# Right now it is changed manually
select_col = False

dataIS = 'EEG'

# Use this function to enter the type of data you want analyzed
# call_get_data = False
# while not call_get_data:
#     dataIS = input('Enter type of Data (NIRS or EEG): ')
#     if dataIS == 'NIRS' or dataIS == 'EEG':
#         call_get_data = True
        
extract_path = f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\{dataIS}_Data'
read_path = f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\{dataIS}_Data\\All_Data\\All_Data.csv'

try:
    f = open(read_path)
    all_data_file_exists = True
    f.close()
except FileNotFoundError:
    print("Could not open")
        
print(all_data_file_exists)
# Reading the information from the csv file into the dataset
if not all_data_file_exists:
    # r=root, d=directories, f = files
    for r, d, f in os.walk(extract_path):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r, file))
    for i in range(len(files)):
        df = pd.read_csv(files[i])    
        #* Data must be divided into attributes and labels
        X = df.iloc[:,:].to_numpy()
        
        #! Remove time -10 to 0
        #Remove the data from -10 sec to 0 sec
        # if dataIS =='NIRS':
        
        c=0
        for val in df['Time']:
            c+=1
            if val < 0:
                df  = df[df.index != c]
        
        # T = []
        # c=0
        # for t in X[0]:
        #     if c+rows_avg <= len(X[0]):
        #         if round(t,2) == round(X[0,c+rows_avg],2):
        #             T.append('')
        #         else:
        #             T.append(f'{round(t,2)}-{round(X[0,c+rows_avg],2)}')
        #     else:
        #         T.append('')
        # print(len(T))
        
        Y=[]
        for _ in range(len(X)):
            if '0-back' in files[i]:
                Y.extend([0])
            elif '2-back' in files[i]:
                Y.extend([2])
            elif '3-back' in files[i]:
                Y.extend([3])
            else:
                continue
        
        # Make an average of N rows
        # sliding window use: for j in range(0,len(X))
        # non-sliding average use: for j in range(0,len(X),rows_avg)
        for j in range(0,len(X),rows_avg):
            # TODO : Add the title of each column somehow
            df2 = pd.DataFrame([np.mean(X[j:j+rows_avg,1:], axis=0)])
            # if len(T) == len(Y):
            #     df2.insert(0,'Time',T[j])
            df2.insert(0,'Y',Y[j])
            hold = hold.append(df2, ignore_index = True)

    # Removes the NaN from EEG and NIRS
    if hold.isnull().values.any() and dataIS =='EEG':
        for i in range(hold.shape[0]):
            if hold.isna().any(axis=1)[i]:
                hold  = hold[hold.index != i]
    
    
    if np.isnan(hold.iloc[2,hold.shape[1]-1]) and dataIS =='NIRS':
        print(hold.head())
        del hold[f'{hold.shape[1]-2}']
        print('here')
    
    hold.to_csv(read_path, index=False)

# Check if we are restricting our data to just the selected columns
if select_col:
    h = pd.read_csv(read_path)
    hold.insert(0,'Y',h['Y'])
    df = pd.read_csv(f'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro\\{dataIS}_Data\\VP001-{dataIS}\\0-back_session\\data_0-back_session_trial001.csv')
    col_list = []
    c = 0
    for col in df.columns:
        if 'S1D1'in col or 'S13D13' in col:
            if 'HbT' in col:
                c+=1
            else:
                hold.insert(len(hold.columns),df.columns[c],h[f'{c}'])
                c +=1
        else:
            c+=1
else:
    hold = pd.read_csv(read_path)

# Removes the NaN from NIRS
if np.isnan(hold.iloc[2,hold.shape[1]-1]):
    # print(hold.head())
    del hold[f'{hold.shape[1]-2}']
    print('here')

print(hold.head())
print(hold.shape)

# hold.shape[1]-1
X = hold.iloc[:,2:].to_numpy()
Y = hold.iloc[:,0].to_numpy()
# print(X[:2])
# print(Y[:2])


def random_forest():
    rf_accuracy = 0
    
    for _ in range(count): 
        # Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        # sc = StandardScaler()
        # x_train = sc.fit_transform(x_train)
        # x_test = sc.fit_transform(x_test)
        # 3d scaler
        # scalers = {}
        # for i in range(x_train.shape[1]):
        #     scalers[i] = StandardScaler()
        #     x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :]) 

        # for i in range(x_test.shape[1]):
        #     x_test[:, i, :] = scalers[i].transform(x_test[:, i, :]) 
        
        #Random Forest Model
        classifier = RandomForestClassifier(n_estimators=500,random_state = 0)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        rf_accuracy += accuracy_score(y_test, y_pred)

    rf_accuracy = rf_accuracy/count

    string =("--------------------------------Evaluating Random Forest--------------------------------\n")
    #print('Prediction: ', y_pred)
    #print('Actual:     ', y_test)
    string +=(f'Accuracy: {rf_accuracy}\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {y_pred[:4]}\n')
    
    return (string)

def SVM():
    svm_accuracy = 0

    for _ in range(count): 
        #Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        # sc = StandardScaler()
        # x_train = sc.fit_transform(x_train)
        # x_test = sc.fit_transform(x_test)
        
        #SVM
        model = svm.SVC()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        svm_accuracy += accuracy_score(y_test, prediction)

    svm_accuracy = svm_accuracy/count

    string =("--------------------------------Evaluating SVM--------------------------------\n")
    #print('Prediction: ', prediction)
    #print('Actual:     ', y_test)
    string +=(f'Accuracy: {svm_accuracy}\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def knn():
    knn_accuracy = 0

    for _ in range(count): 
        #Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        # sc = StandardScaler()
        # x_train = sc.fit_transform(x_train)
        # x_test = sc.fit_transform(x_test)
        
        #KNN
        #TODO Play around with the amount fo neighbors
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        knn_accuracy += accuracy_score(y_test, prediction)

    knn_accuracy = knn_accuracy/count

    string =("--------------------------------Evaluating KNN--------------------------------\n")
    #print('Prediction: ', prediction)
    #print('Actual:     ', y_test)
    string +=(f'Accuracy: {knn_accuracy}\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def NN():
    accuracy = 0

    for _ in range(count):
        #Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        
        #NN
        #TODO Play around with the different settings
        #Use adam for large datasets, and lbfgs for smaller datasets
        
        model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 100),max_iter = 1000)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        accuracy += accuracy_score(y_test, prediction)

    accuracy = accuracy/count

    string =("--------------------------------Evaluating Neural Network--------------------------------\n")
    #print('Prediction: ', prediction)
    #print('Actual:     ', y_test)
    string +=(f'Accuracy: {accuracy}\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def gaussNB():
    accuracy = 0

    for _ in range(count): 
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        
        #Gauss
        #TODO Play around with the amount fo neighbors
        model = GaussianNB()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        accuracy += accuracy_score(y_test, prediction)

    accuracy = accuracy/count

    string =("--------------------------------Evaluating Naive Bayes--------------------------------\n")
    #print('Prediction: ', prediction)
    #print('Actual:     ', y_test)
    string +=(f'Accuracy: {accuracy}\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)
    
# if __name__ == '__main__':
#     print(random_forest())
#     print(SVM())
#     print(knn())
#     print(NN())
#     print(gaussNB())
    # Play a sound when the code finishes
    # winsound.Beep(freq, duration)
    # print(f"X: {X[:2]}")
    # print(f"Y: {Y[:2]}")
    # print(X.shape)
    # print(Y.shape)    

    