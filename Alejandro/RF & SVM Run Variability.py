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

# dataset = r'C:\Users\owner\Documents\GitHub\Capstone_490\Alejandro\TrainingData\Merged\Merged.csv'
path = r'C:\Users\owner\Documents\GitHub\Capstone_490\Alejandro\TrainingData\Merged'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))
    
print(files)
# print(os.path.join(dirpath, name))

rows_avg = 10
hold = pd.DataFrame()

# Reading the information from the csv file into the dataset
for i in range(len(files)):
    df = pd.read_csv(files[i])

    # print(df.head())
    for col in df.columns:
        if 'Source.Name' in col or 'Time' in col or 'S1D1'in col or 'S13D13' in col:
            if 'HbT' in col:
                del df[f'{col}']
            else:
                continue
        else:
            del df[f'{col}']
        
    rows = df.shape[0]
    columns = df.shape[1]
    # print(df.head())

    # Prepearing the Data for training
    #* Data must be divided into attributes and labels
    X = df.iloc[:,:].to_numpy()
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

    # # Make an average of N rows
    for j in range(0,len(X),rows_avg):
        df2 = pd.DataFrame([np.mean(X[j:j+5,1:], axis=0)])
        df2.insert(0,'Y',Y[j])
        hold = hold.append(df2, ignore_index = True)
        
        # if count == 5:
        #     break
        # count +=1

print(hold.head())

X = hold.iloc[:,1:].to_numpy()
Y = hold.iloc[:,0].to_numpy()
print(X[:2])
print(Y[:2])


le = preprocessing.LabelEncoder()
le.fit(hold['Y'])
Y = le.fit_transform(Y)

count = 10

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
        classifier = RandomForestClassifier(n_estimators=20,random_state = 0)
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
        model = KNeighborsClassifier(n_neighbors=3)
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
        
        model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(50, 20),max_iter = 1000)
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
    
if __name__ == '__main__':
    print(random_forest())
    print(SVM())
    print(knn())
    print(NN())
    print(gaussNB())
    # print(f"X: {X[:2]}")
    # print(f"Y: {Y[:2]}")
    # print(X.shape)
    # print(Y.shape)    

    