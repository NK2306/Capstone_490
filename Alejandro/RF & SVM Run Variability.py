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
from itertools import islice

def exec_Code(dataIS, row_name, window_analysis):
    #Constants
    rows_avg = 10
    count = 1
    files = []
    all_data_file_exists = False
    select_col = False
    window_size = 5
    
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
    
    # Reading the information from the csv file into the dataset
    if not all_data_file_exists:
        scrape_files(extract_path, read_path, rows_avg, dataIS)

    # Check if we are restricting our data to just the selected columns
    hold = check_columns(extract_path,read_path, select_col, dataIS)
    
    #* Sliding window analysis to get accuracy in that time frame
    #TODO Does not work yet
    # if window_analysis:
    #     window(window_size, read_path, select_col, count)
    
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
    X = hold.iloc[:,1:].to_numpy()
    Y = hold.iloc[:,0].to_numpy()
    print(X)
    print(Y)
    run_models(count, X, Y, select_col)
    
def check_columns(extract_path,read_path, select_col, dataIS,):
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
    
    return hold

def window(window_size, read_path, select_col, count):
    rf_accuracy = []
    svm_accuracy = []
    knn_accuracy = []
    nn_accuracy = []
    gauss_accuracy = []
    consensus_accuracy = []
    acc_values = []
    
    
    h = pd.read_csv(read_path,nrows = window_size)
    
    for i in range(h.shape[0]):
        hold = pd.read_csv(read_path,skiprows = i,nrows = window_size)
        hold.columns = h.columns
        X = hold.iloc[:,1:].to_numpy()
        Y = hold.iloc[:,0].to_numpy()
        #! Cannot have only one type of data analyzed to make the prediction.
        #! If we restrict the rows then we need proper representation on the time axis from all the types of data gathered [0,2,3]
        run_models(count, X, Y, select_col)
        # print(acc_values[1])
        
        
      
def scrape_files(extract_path,read_path, rows_avg, dataIS):
    hold = pd.DataFrame()
    files = []
    
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
        column_names = df.columns
        col_name = []
        
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
        
        if 'Time' in column_names:
            for i in range(len(column_names)):
                if column_names[i] != 'Time':
                    col_name.append(column_names[i])               
                    
            for val in df['Time']:
                c+=1
                if val < 0:
                    df  = df[df.index != c]
        
        # Make an average of N rows
        # sliding window use: for j in range(0,len(X))
        # non-sliding average use: for j in range(0,len(X),rows_avg)
        for j in range(0,len(X),rows_avg):
            # TODO : Add the title of each column somehow
            df2 = pd.DataFrame([np.mean(X[j:j+rows_avg,1:], axis=0)], columns=col_name)
            df2.insert(0,'Y',Y[j])
            hold = hold.append(df2, ignore_index = True)

    # Removes the NaN from EEG and NIRS
    if hold.isnull().values.any() and dataIS =='EEG':
        for i in range(hold.shape[0]):
            if hold.isna().any(axis=1)[i]:
                hold  = hold[hold.index != i]
    
    
    hold.to_csv(read_path, index=False)
    
def run_models(numOfIter, X, Y, select_col):
    
    rf_accuracy = 0
    svm_accuracy = 0
    knn_accuracy = 0
    nn_accuracy = 0
    gauss_accuracy = 0
    consensus_accuracy = 0
    string = ''
    
    for _ in range(numOfIter): 
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

    rf_accuracy = (rf_accuracy/numOfIter)*100

    string +=("--------------------------------Evaluating Random Forest--------------------------------\n")
    string +=(f'Accuracy: {rf_accuracy} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {rf_prediction[:4]}\n')

    svm_accuracy = (svm_accuracy/numOfIter)*100
    
    string +=("--------------------------------Evaluating SVM--------------------------------\n")
    string +=(f'Accuracy: {svm_accuracy} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {svm_prediction[:4]}\n')

    knn_accuracy = (knn_accuracy/numOfIter)*100

    string +=("--------------------------------Evaluating KNN--------------------------------\n")
    string +=(f'Accuracy: {knn_accuracy} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {knn_prediction[:4]}\n')    
    
    nn_accuracy = (nn_accuracy/numOfIter)*100

    string +=("--------------------------------Evaluating Neural Network--------------------------------\n")
    string +=(f'Accuracy: {nn_accuracy} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {nn_prediction[:4]}\n')
    
    gauss_accuracy = (gauss_accuracy/numOfIter)*100

    string +=("--------------------------------Evaluating Naive Bayes--------------------------------\n")
    string +=(f'Accuracy: {gauss_accuracy} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {gauss_prediction[:4]}\n')
    
    consensus_accuracy = (consensus_accuracy/numOfIter)*100

    string +=("--------------------------------Evaluating Consensus Accuracy--------------------------------\n")
    string +=(f'Accuracy: {consensus_accuracy} %\n')
    string +=(f'y_test: {y_test[:4]}\nPrediction: {consensus_prediction[:4]}\n')
    
    print(string)
    
    return [rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy, consensus_accuracy]

if __name__ == '__main__':
    # x is data type (EEG or NIRS), y is boolean (restrict data or not)
    # x = str(sys.argv[1])
    # y = str(sys.argv[2])
    # rows_names = []
    x = 'EEG'
    y = ''
    z = 'True'
    
    if z == 'True':
        window_analysis = True
    elif z == 'False':
        window_analysis = False
    
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
        
    sys.stdout.write(str(exec_Code(x, rows_names, window_analysis)))