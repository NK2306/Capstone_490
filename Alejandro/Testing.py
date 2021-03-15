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

def exec_Code(dataIS, row_name):
    #Constants
    rows_avg = 10
    count = 10
    files = []
    all_data_file_exists = False
    rf_accuracy = 0
    svm_accuracy = 0
    knn_accuracy = 0
    nn_accuracy = 0
    gauss_accuracy = 0
    consensus_accuracy = 0
    string = ''
    select_col = False
    
    if row_name != '':
        select_col = True
    
    
    hold = pd.DataFrame()
            
    extract_path = f'{dataIS}_Data'
    read_path = f'{dataIS}_Data\\All_Data\\All_Data.csv'

    try:
        f = open(read_path)
        all_data_file_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
            
    #print(all_data_file_exists)
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
        
        
        hold.to_csv(read_path, index=False)

    # Check if we are restricting our data to just the selected columns
    if select_col:
        h = pd.read_csv(read_path)
        hold.insert(0,'Y',h['Y'])
        
        col_list = []
        c = 0
        
        if dataIS == "NIRS":   
            df = pd.read_csv(f'{extract_path}\\VP001-{dataIS}\\0-back_session\\data_0-back_session_trial001.csv')
            for col in df.columns:
                # print(col)
                # print(c)
                for r in range(len(row_name)):
                    # print(row_name[r])
                    if row_name[r] == col:
                        if 'HbO' in col or 'HbR' in col:
                            hold.insert(len(hold.columns),df.columns[c],h[f'{c}'])
                c+=1
                
        elif dataIS == "EEG":
            df = pd.read_csv(f'{extract_path}\\ERP_VP001_0-back.csv')
            for col in df.columns:
                # print(col)
                # print(c)
                for r in range(len(row_name)):
                    # print(row_name[r])
                    if row_name[r] == col:
                        hold.insert(len(hold.columns),df.columns[c],h[f'{c}']) 
                c +=1
            


    else:
        hold = pd.read_csv(read_path)

    # Removes the NaN from NIRS
    if np.isnan(hold.iloc[2,hold.shape[1]-1]) and dataIS =='NIRS':
        # print(hold.head())
        del hold[f'{hold.shape[1]-2}']
        #print('here')

    print(f"Data type: {dataIS}")
    print(f"Restrict Columns: {select_col}\n")
    print("------------------------First five rows of Dataset--------------------------")
    print(hold.head())
    print()
    X = hold.iloc[:,2:].to_numpy()
    Y = hold.iloc[:,0].to_numpy()
    
    for _ in range(count): 
        # Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=500,random_state = 0)
        rf_model.fit(x_train, y_train)
        rf_prediction = rf_model.predict(x_test)
        rf_accuracy += accuracy_score(y_test, rf_prediction)
        
        # SVM
        svm_model = svm.SVC()
        svm_model.fit(x_train, y_train)
        svm_prediction = svm_model.predict(x_test)
        svm_accuracy += accuracy_score(y_test, svm_prediction)
        
        # KNN
        # TODO Play around with the amount fo neighbors
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_train, y_train)
        knn_prediction = knn_model.predict(x_test)
        knn_accuracy += accuracy_score(y_test, knn_prediction)

        # NN
        #TODO Play around with the different settings
        # Use adam for large datasets, and lbfgs for smaller datasets
        if select_col:
            solver_string = 'lbfgs'
        else:
            solver_string = 'adam'
        #* StandardScaler is ESSENTIAL for NN
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
            
        nn_model = MLPClassifier(solver= solver_string, alpha=1e-5, hidden_layer_sizes=(200, 100),max_iter = 1000)
        nn_model.fit(x_train, y_train)
        nn_prediction = nn_model.predict(x_test)
        nn_accuracy += accuracy_score(y_test, nn_prediction)
        
        # Gauss
        # TODO Play around with the amount fo neighbors
        gauss_model = GaussianNB()
        gauss_model.fit(x_train, y_train)
        gauss_prediction = gauss_model.predict(x_test)
        gauss_accuracy += accuracy_score(y_test, gauss_prediction)
        
        # print(rf_model.predict(x_test)[0])
        
        consensus_prediction = nn_model.predict(x_test)
        
        #* Check highest consensus of two models as a whole
        # Percentage similarity of lists 
     
        # nn_rf = len(set(nn_prediction) & set(rf_prediction)) / float(len(set(nn_prediction) | set(rf_prediction))) * 100
        # nn_knn = len(set(nn_prediction)& set(knn_prediction)) / float(len(set(nn_prediction) | set(knn_prediction))) * 100
        # knn_rf = len(set(knn_prediction)& set(rf_prediction)) / float(len(set(knn_prediction) | set(rf_prediction))) * 100
        # nn_svm = len(set(nn_prediction)& set(svm_prediction)) / float(len(set(nn_prediction) | set(svm_prediction))) * 100
        # rf_svm = len(set(svm_prediction)& set(rf_prediction)) / float(len(set(svm_prediction) | set(rf_prediction))) * 100
        # knn_svm = len(set(knn_prediction)& set(svm_prediction)) / float(len(set(knn_prediction) | set(svm_prediction))) * 100
        
        # acc_list = [nn_rf,nn_knn,knn_rf,nn_svm,rf_svm,knn_svm]
        # acc_list.sort()
        
        # # Consensus between Neural Network and Random Forest
        # if acc_list[-1] == nn_rf:
        #     consensus_prediction = nn_prediction
        #     print('Consensus between NN & RF')
                
        # # Consensus between Neural Network and KNN
        # elif acc_list[-1] == nn_knn:
        #     consensus_prediction = nn_prediction
        #     print('Consensus between NN & KNN')
            
        # # Consensus between KNN and Random Forest
        # elif acc_list[-1] == knn_rf:
        #     consensus_prediction = rf_prediction
        #     print('Consensus between KNN & RF')
            
        # # Consensus between Neural Network and SVM    
        # elif acc_list[-1] == nn_svm:
        #     consensus_prediction = nn_prediction
        #     print('Consensus between NN & SVM')
            
        # # Consensus between SVM and Random Forest    
        # elif acc_list[-1] == rf_svm:
        #     consensus_prediction = rf_prediction
        #     print('Consensus between SVM & RF')
            
        # # Consensus between KNN and SVM
        # elif acc_list[-1] == knn_svm:
        #     consensus_prediction= knn_prediction
        #     print('Consensus between KNN & SVM')
            
        # # Else default to NN (impossible edge case)
        # else:
        #     consensus_prediction = nn_prediction
        #     print('No Consensus')
        
        # #* Checks consensus individually
        for i in range(len(nn_prediction)):
            # Consensus between Neural Network and Random Forest
            if nn_prediction[i] == rf_prediction[i]:
                consensus_prediction[i] = nn_prediction[i]
                
            # Consensus between Neural Network and KNN
            elif nn_prediction[i] == knn_prediction[i]:
                consensus_prediction[i] = nn_prediction[i]
                
            # Consensus between KNN and Random Forest
            elif rf_prediction[i] == knn_prediction[i]:
                consensus_prediction[i] = rf_prediction[i]
                
            # Consensus between Neural Network and SVM    
            elif nn_prediction[i] == svm_prediction[i]:
                consensus_prediction[i]= nn_prediction[i]
                
            # Consensus between SVM and Random Forest    
            elif rf_prediction[i] == svm_prediction[i]:
                consensus_prediction[i] = rf_prediction[i]
                
            # Consensus between KNN and SVM
            elif knn_prediction[i] == svm_prediction[i]:
                consensus_prediction[i] = knn_prediction[i]
                
            # Else default to NN
            else:
                consensus_prediction[i] = nn_prediction[i]
                
                
                
        consensus_accuracy += accuracy_score(y_test,consensus_prediction)

    rf_accuracy = rf_accuracy/count

    string +=("--------------------------------Evaluating Random Forest--------------------------------\n")
    string +=(f'Accuracy: {rf_accuracy*100} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {rf_prediction[:4]}\n')

    svm_accuracy = svm_accuracy/count
    
    string +=("--------------------------------Evaluating SVM--------------------------------\n")
    string +=(f'Accuracy: {svm_accuracy*100} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {svm_prediction[:4]}\n')

    knn_accuracy = knn_accuracy/count

    string +=("--------------------------------Evaluating KNN--------------------------------\n")
    string +=(f'Accuracy: {knn_accuracy*100} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {knn_prediction[:4]}\n')    
    
    nn_accuracy = nn_accuracy/count

    string +=("--------------------------------Evaluating Neural Network--------------------------------\n")
    string +=(f'Accuracy: {nn_accuracy*100} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {nn_prediction[:4]}\n')
    
    gauss_accuracy = gauss_accuracy/count

    string +=("--------------------------------Evaluating Naive Bayes--------------------------------\n")
    string +=(f'Accuracy: {gauss_accuracy*100} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {gauss_prediction[:4]}\n')
    
    consensus_accuracy = consensus_accuracy/count

    string +=("--------------------------------Evaluating Consensus Accuracy--------------------------------\n")
    string +=(f'Accuracy: {consensus_accuracy*100} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {consensus_prediction[:4]}\n')
    
    print(string)



def random_forest(X, Y):
    rf_accuracy = 0
    
    for _ in range(count): 
        # Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # Random Forest Model
        classifier = RandomForestClassifier(n_estimators=500,random_state = 0)
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        rf_accuracy += accuracy_score(y_test, prediction)

    rf_accuracy = rf_accuracy/count

    string =("--------------------------------Evaluating Random Forest--------------------------------\n")
    string +=(f'Accuracy: {rf_accuracy*100}%\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def SVM(X, Y):
    svm_accuracy = 0

    for _ in range(count): 
        # Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # SVM
        model = svm.SVC()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        svm_accuracy += accuracy_score(y_test, prediction)

    svm_accuracy = svm_accuracy/count

    string =("--------------------------------Evaluating SVM--------------------------------\n")
    string +=(f'Accuracy: {svm_accuracy*100}%\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def knn(X, Y):
    knn_accuracy = 0

    for _ in range(count): 
        # Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # KNN
        # TODO Play around with the amount fo neighbors
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        knn_accuracy += accuracy_score(y_test, prediction)

    knn_accuracy = knn_accuracy/count

    string =("--------------------------------Evaluating KNN--------------------------------\n")
    string +=(f'Accuracy: {knn_accuracy*100}%\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def NN(X, Y):
    nn_accuracy = 0

    for _ in range(count):
        # Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        
        # NN
        #TODO Play around with the different settings
        # Use adam for large datasets, and lbfgs for smaller datasets
        model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 100),max_iter = 1000)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        nn_accuracy += accuracy_score(y_test, prediction)

    nn_accuracy = nn_accuracy/count

    string =("--------------------------------Evaluating Neural Network--------------------------------\n")
    string +=(f'Accuracy: {nn_accuracy*100}%\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def gaussNB(X, Y):
    
    gauss_accuracy = 0

    for _ in range(count): 
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        # sc = StandardScaler()
        # x_train = sc.fit_transform(x_train)
        # x_test = sc.fit_transform(x_test)
        
        # Gauss
        # TODO Play around with the amount fo neighbors
        model = GaussianNB()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        gauss_accuracy += accuracy_score(y_test, prediction)

    gauss_accuracy = gauss_accuracy/count

    string =("--------------------------------Evaluating Naive Bayes--------------------------------\n")
    string +=(f'Accuracy: {gauss_accuracy*100}%\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {prediction[:4]}\n')
    
    return (string)

def consensus_score(X, Y):
    
    accuracy = 0

    for _ in range(count): 
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        
        # Consensus of Neural Network & Random Forest
        rf_classifier = RandomForestClassifier(n_estimators=500,random_state = 0)
        rf_classifier.fit(x_train, y_train)
        nn_model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 100),max_iter = 1000)
        nn_model.fit(x_train, y_train)
        if rf_classifier.predict(x_test) == nn_model.predict(x_test):
            prediction = rf_classifier.predict(x_test) 
        consensus_accuracy += accuracy_score(y_test, rf_prediction)
        
        

    gauss_accuracy = gauss_accuracy/count

    string =("--------------------------------Evaluating Consensus Accuracy--------------------------------\n")
    string +=(f'Accuracy: {gauss_accuracy*100}%\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {rf_prediction[:4]}\n')
    
    return (string)

if __name__ == '__main__':
    # x is data type (EEG or NIRS), y is boolean (restrict data or not)
    # x = str(sys.argv[1])
    # y = str(sys.argv[2])
    # rows_names = []
    x = 'EEG'
    y = ''
    
    print(y)
    
    if y == 'frontal':
        if x == "EEG":
            rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']
    
    elif y == 'parietal':
        if x == "EEG":
            rows_names=  ['FC5','FC1','T7','C3','Cz','CP5','CP1','FC2','FC6','C4','T8','CP2','CP6','P4']
            
    elif y == 'occipital':
        if x == "EEG":
            rows_names=  ['P7','P3','Pz', 'P4','P8', 'POz', 'O1','O2']
    else:
        rows_names = y
            
    
    all_data_file_exists = False
        
    sys.stdout.write(str(exec_Code(x, rows_names)))