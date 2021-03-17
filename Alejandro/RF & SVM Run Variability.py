import sys
import pandas as pd
import pickle
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
from sklearn.model_selection import cross_val_score # pylint: disable=import-error
from sklearn.pipeline import make_pipeline
# from keras.models import Sequential
# from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from itertools import islice

def exec_Code(dataIS, row_name, window_analysis,tts_participant):
    #Constants
    rows_avg = 10
    count = 5
    files = []
    all_data_file_exists = False
    select_col = False
    window_size = 100
    # Boolean
    train_test_split_participants = tts_participant
    
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
    if select_col:
        hold = check_columns(extract_path,read_path, row_name, dataIS)
    else:
        hold = pd.read_csv(read_path)
        
    
    # Removes the NaN from NIRS
    if np.isnan(hold.iloc[2,hold.shape[1]-1]) and dataIS =='NIRS':
        # print(hold.head())
        del hold[f'{hold.shape[1]-2}']
        #print('here')

    print(f"Data type: {dataIS}")
    print(f"Restrict Columns: {select_col}\n")
    
    #* Sliding window analysis to get accuracy in that time frame
    if window_analysis:
        window(window_size, extract_path, select_col, count, dataIS)
    
    elif train_test_split_participants:
        # Train the model with the first 20 participants
        if dataIS == "EEG":
            #60%
            for participant in range(1,4):
                if participant<10:
                    participant = f'0{participant}'
                    # print(participant)
                    acc_val = train_models(extract_path, participant,select_col,dataIS)
                    print("Participant: ", participant)
                    print(acc_val[-1])
                else:
                    acc_val = train_models(extract_path, participant,select_col,dataIS)
                    print("Participant: ", participant)
                    print(acc_val[-1])
            
            #40%
            for participant in range(4,6):  
                # Test the model witht the last 2 participants
                if participant <10:
                    participant = f'0{participant}'
                    acc_val = test_models(extract_path, participant,select_col,dataIS)
                    print("Participant: ", participant)
                    print(acc_val[-1])
                else:
                    acc_val = test_models(extract_path, participant,select_col,dataIS)
                    print("Participant: ", participant)
                    print(acc_val[-1])
       
        elif dataIS == "NIRS":
            
            last_train_particip = 20
            acc_val = train_models(extract_path, dataIS, last_train_particip)
            # print(acc_val[-1])
                
    
            # Test the model witht the last 6 participants
            first_test_participant = last_train_particip
            acc_val = test_models(extract_path,dataIS,first_test_participant)
            # print(acc_val[-1])
                
    else:
        print("------------------------First five rows of Dataset--------------------------")
        print(hold.head())
        print()

        for participant in range(1,4):
            participant = f'0{participant}'
            print("Train:", participant)
            
        for participant in range(4,6):
            print("Test:", participant)
        

        # X = hold.iloc[:,1:].to_numpy()
        # Y = hold.iloc[:,0].to_numpy()
        # print(cross_validate(count,X, Y, select_col))
        # print(acc_val[-1])
    
def check_columns(extract_path, read_path, row_name, dataIS,):
    
    h = pd.read_csv(read_path)
    hold = pd.DataFrame()
    hold.insert(0,'Y',h['Y'])
    
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
                        hold.insert(len(hold.columns),df.columns[c],h[f'{df.columns[c]}'])
            c+=1
            
    elif dataIS == "EEG":
        # df = pd.read_csv(f'{extract_path}\\ERP_VP001_0-back.csv')
        for col in h.columns:
            # print(col)
            # print(c)
            for r in range(len(row_name)):
                # print(row_name[r])
                if row_name[r] == col:
                    hold.insert(len(hold.columns),col,h[f'{col}']) 
            c +=1      

    return hold

def scrape_files(extract_path,read_path, rows_avg, dataIS):
    
    hold = pd.DataFrame()
    files = []
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(extract_path):
        for file in f:
            if '_average_' in file:
                continue
            
            elif '.csv' in file:
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
                
    if 'Unnamed' in hold.columns[-1] and dataIS =='NIRS':
        del hold['Unnamed: 109']
        print(hold.head())
    
    hold.to_csv(read_path, index=False)

def window(window_size, extract_path, select_col, count,dataIS):
    print("window")
    skip_num = 5
    
    if dataIS == "EEG":
        h = pd.read_csv(f"{extract_path}\\ERP_VP001_0-back.csv")
        for j in range(1,6):
            time_frame = []
            rf_accuracy = []
            svm_accuracy = []
            knn_accuracy = []
            nn_accuracy = []
            gauss_accuracy = []
            consensus_accuracy = []
            print('Participant:', j)
            df = pd.DataFrame()

            
            for i in range(int(h.shape[0]/skip_num)): #h.shape[0]
                df_0_back = pd.read_csv(f"{extract_path}\\ERP_VP0{j}_0-back.csv", skiprows = i+skip_num, nrows = window_size)
                Y_0_back = [0]*df_0_back.shape[0]
                df_0_back.columns = h.columns
                df_0_back.insert(0,'Y',Y_0_back)
                time_frame.append("{:.3f}".format(np.mean(df_0_back['Time'])))
                # print ('time frame:')
                # print(time_frame)
                
                df_2_back = pd.read_csv(f"{extract_path}\\ERP_VP0{j}_2-back.csv", skiprows = i+skip_num, nrows = window_size)
                Y_2_back = [2]*df_2_back.shape[0]
                df_2_back.columns = h.columns
                df_2_back.insert(0,'Y',Y_2_back)
                                
                df_3_back = pd.read_csv(f"{extract_path}\\ERP_VP0{j}_3-back.csv", skiprows = i+skip_num, nrows = window_size)
                Y_3_back = [3]*df_3_back.shape[0]
                df_3_back.columns = h.columns
                df_3_back.insert(0,'Y',Y_3_back)
                
                hold = df_0_back.copy()
                hold = hold.append(df_2_back, ignore_index = True)
                hold = hold.append(df_3_back, ignore_index = True)
                # print('Hold: ')
                # print(hold.head())
            
                X = hold.iloc[:,2:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_values = run_models(count, X, Y, select_col)
                rf_accuracy.append(acc_values[0])
                svm_accuracy.append(acc_values[1])
                knn_accuracy.append(acc_values[2])
                nn_accuracy.append(acc_values[3])
                gauss_accuracy.append(acc_values[4])
                consensus_accuracy.append(acc_values[5])
            
            df.insert(0,'Random Forest',rf_accuracy)
            df.insert(1,'SVM',svm_accuracy)
            df.insert(2,'KNN',knn_accuracy)
            df.insert(3,'NN',nn_accuracy)
            df.insert(4,'Gauss',gauss_accuracy)
            df.insert(5,'Consensus',consensus_accuracy)
            df.insert(6,'Time', time_frame)
            lines = df.plot.line(x='Time')
            lines.set_title(f'Sliding window Performance Participant {j} EEG')
            lines.set_ylabel('Accuracy %')
            
            # plt.show()
            plt.savefig(f'Sliding_EEG\\Sliding window Performance Participant {j} EEG.png')
    
    elif dataIS == "NIRS":
        
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(extract_path):
            for file in f:
                if 'average_210315_1325.csv' in file:
                    files.append(os.path.join(r, file))
            
        # print(files)
        h = pd.read_csv(files[0])
        if 'Unnamed' in h.columns[-1]:
            del h['Unnamed: 109']
            print(h.head())
        
        new_files = []
        
        i = 0
        while len(files)>i:
            
            participant = f'{files[i][files[i].find("VP0")+3]}{files[i][files[i].find("VP0")+4]}'
            # print('Participant:', participant)
            
            if f"VP0{participant}" in files[i]:
               
                if f"VP0{participant}" in files[i] and f"VP0{participant}" in files[i+1] and f"VP0{participant}" in files[i+2]:
                    # print(files[i])
                    # print(files[i+1])
                    # print(files[i+2])
                    new_files.append(files[i])
                    new_files.append(files[i+1])
                    new_files.append(files[i+2])
                    i+=1
                    i+=1
                    i+=1
                    continue
                                    
                elif f"VP0{participant}" in files[i] and f"VP0{participant}" in files[i+1] and f"VP0{participant}" not in files[i+2]:
                    # print(files[i])
                    # print(files[i+1])
                    i+=1
                    i+=1
                    continue
                
                elif f"VP0{participant}" in files[i] and f"VP0{participant}" not in files[i+1] and f"VP0{participant}" not in files[i+2]:
                    # print(files[i])
                    i+=1
                    continue
            i+=1
        
        three = 3
        
        for i in range(5*three,10*three,three): #len(new_files)
            df = pd.read_csv(new_files[i])
            time_frame = []
            rf_accuracy = []
            svm_accuracy = []
            knn_accuracy = []
            nn_accuracy = []
            gauss_accuracy = []
            consensus_accuracy = []
            make_png = False
            c=0
            participant = f'{files[i][files[i].find("VP0")+3]}{files[i][files[i].find("VP0")+4]}'
            print('Participant:', participant)
            
            if f"VP0{participant}" in new_files[i]:
                print(new_files[i])
                print(new_files[i+1])
                print(new_files[i+2])
                
            df = pd.DataFrame()

            for j in range(int(h.shape[0]-10)): #h.shape[0]
                
                df_0_back = pd.read_csv(new_files[i], skiprows = j+skip_num, nrows = window_size)
                Y_0_back = [0]*df_0_back.shape[0]
                df_0_back.columns = h.columns
                df_0_back.insert(0,'Y',Y_0_back)
                time_frame.append("{:.3f}".format(np.mean(df_0_back['Time'])))
                # print ('time frame:')
                # print(time_frame)
            
                df_2_back = pd.read_csv(new_files[i+1], skiprows = j+skip_num, nrows = window_size)
                Y_2_back = [2]*df_2_back.shape[0]
                df_2_back.columns = h.columns
                df_2_back.insert(0,'Y',Y_2_back)
                    
                df_3_back = pd.read_csv(new_files[i+2], skiprows = j+skip_num, nrows = window_size)
                Y_3_back = [3]*df_3_back.shape[0]
                df_3_back.columns = h.columns
                df_3_back.insert(0,'Y',Y_3_back)
            
                hold = df_0_back.copy()
                hold = hold.append(df_2_back, ignore_index = True)
                hold = hold.append(df_3_back, ignore_index = True)
                # print('Hold: ')
                # print(hold.head())
            
                X = hold.iloc[:,2:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                # print(Y)
                acc_values = cross_validate(count, X, Y, select_col)
                rf_accuracy.append(acc_values[0])
                svm_accuracy.append(acc_values[1])
                knn_accuracy.append(acc_values[2])
                # nn_accuracy.append(acc_values[3])
                gauss_accuracy.append(acc_values[3])
                # consensus_accuracy.append(acc_values[5])
            
            print(hold.head())
            
            df.insert(0,'Random Forest',rf_accuracy)
            df.insert(1,'SVM',svm_accuracy)
            df.insert(2,'KNN',knn_accuracy)
            # df.insert(3,'NN',nn_accuracy)
            df.insert(3,'Gauss',gauss_accuracy)
            # df.insert(5,'Consensus',consensus_accuracy)
            df.insert(4,'Time', time_frame)
            lines = df.plot.line(x='Time')
            lines.set_title(f'Sliding window Performance Participant {participant} NIRS')
            lines.set_ylabel('Accuracy %')
            # print(time_frame)
            for t in range(len(time_frame)):
                if -0.1 <= float(time_frame[t]) <= 0.1:
                    plt.axvline(x=t)
                    break
            
            # plt.show()
            plt.savefig(f'Sliding_NIRS\\Sliding window Performance Participant {participant} NIRS.png')         
                
        #! Cannot have only one type of data analyzed to make the prediction.
        #! If we restrict the rows then we need proper representation on the time axis from all the types of data gathered [0,2,3]
    
def train_models(extract_path,dataIS, last_participant):
    print('train')
    
    df = pd.DataFrame()
    if dataIS =='EEG':
        h = pd.read_csv(f"{extract_path}\\ERP_VP001_0-back.csv")
            
        df_0_back = pd.read_csv(f"{extract_path}\\ERP_VP0{participant}_0-back.csv")
        Y_0_back = [0]*df_0_back.shape[0]
        df_0_back.columns = h.columns
        df_0_back.insert(0,'Y',Y_0_back)
        
        df_2_back = pd.read_csv(f"{extract_path}\\ERP_VP0{participant}_2-back.csv")
        Y_2_back = [2]*df_2_back.shape[0]
        df_2_back.columns = h.columns
        df_2_back.insert(0,'Y',Y_2_back)
                        
        df_3_back = pd.read_csv(f"{extract_path}\\ERP_VP0{participant}_3-back.csv")
        Y_3_back = [3]*df_3_back.shape[0]
        df_3_back.columns = h.columns
        df_3_back.insert(0,'Y',Y_3_back)
        
        hold = df_0_back.copy()
        hold = hold.append(df_2_back, ignore_index = True)
        hold = hold.append(df_3_back, ignore_index = True)
        
        # Removes the NaN from EEG and NIRS
        if hold.isnull().values.any() and dataIS =='EEG':
            for i in range(hold.shape[0]):
                if hold.isna().any(axis=1)[i]:
                    hold  = hold[hold.index != i]
    
        X = hold.iloc[:,2:].to_numpy()
        Y = hold.iloc[:,0].to_numpy()
    
    elif dataIS == "NIRS":
                # df = pd.read_csv(new_files[i])
        files = []
        c=0
        # r=root, d=directories, f = files
        for r, d, f in os.walk(extract_path):
            for file in f:
                if 'average_210315_1325.csv' not in file and 'All_Data.csv' not in file and '.csv' in file:
                    files.append(os.path.join(r, file))
            
        # print(files)
        h = pd.read_csv(files[0])
        if 'Unnamed' in h.columns[-1]:
            del h['Unnamed: 109']
            print(h.head())
        
        hold = pd.DataFrame()

        for i in range(len(files)):
            
            this_participant = f'{files[i][files[i].find("VP0")+3]}{files[i][files[i].find("VP0")+4]}'
            if i+1 != len(files):
                next_participant = f'{files[i+1][files[i+1].find("VP0")+3]}{files[i+1][files[i+1].find("VP0")+4]}'
            elif i+1 == len(files):
                next_participant = ''
            else:
                print("We have an issue")
            
            #if this_participant == to next_participant then collect the data                 
            if this_participant == next_participant: 
                # print('The same')
                # print('This participant:', this_participant)
                # print('Next participant:', next_participant)
                if f'VP0{this_participant}' in files[i] and '0-back' in files[i]:
                    df_0_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_0_back.columns[-1]:
                        del df_0_back['Unnamed: 109']
                    
                    Y_0_back = [0]*df_0_back.shape[0]
                    df_0_back.columns = h.columns
                    df_0_back.insert(0,'Y',Y_0_back)
                    
                    hold = hold.append(df_0_back, ignore_index = True)
                    
                elif f'VP0{this_participant}' in files[i] and '2-back' in files[i]:
                    df_2_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_2_back.columns[-1]:
                        del df_2_back['Unnamed: 109']
                    
                    Y_2_back = [2]*df_2_back.shape[0]
                    df_2_back.columns = h.columns
                    df_2_back.insert(0,'Y',Y_2_back)
                    
                    hold = hold.append(df_2_back, ignore_index = True)
            
                elif f'VP0{this_participant}' in files[i] and '3-back' in files[i]:
                    df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back['Unnamed: 109']
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
            #if this_participant != to next_participant then analyze the data                 
            elif this_participant != next_participant and next_participant != last_participant:
                print('Not the same')
                print(hold.head())
                # print('This participant:', this_participant)
                # print('Next participant:', next_participant)
                X = hold.iloc[:,2:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_val = pickle_model_train(X,Y)
                
                #reseting hold
                hold = pd.DataFrame()
                
            #last_participant reached
            else:
                print('Last Participant:', last_participant)
                print(hold.head())
                X = hold.iloc[:,2:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_val = pickle_model_train(X,Y)
                break
             

    
def pickle_model_train(X,Y):  
    model_exists = False
    string = ''
    # Dividing the data into training and testing sets
    #* Training size is 90% since this is the training set of data, 10% is used for preliminary testing
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
    
    # Random Forest Model
    try:
        f = open("Pickled_models\\rf_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    if model_exists:
        pkl_filename = "Pickled_models\\rf_model.pkl"
        with open(pkl_filename, 'rb') as file:
            rf_model = pickle.load(file)
            rf_model.fit(x_train, y_train)
            rf_prediction = rf_model.predict(x_test)
            rf_accuracy = accuracy_score(y_test, rf_prediction)

    else:
        rf_model = RandomForestClassifier(n_estimators=500)
        rf_model.fit(x_train, y_train)
        rf_prediction = rf_model.predict(x_test)
        rf_accuracy = accuracy_score(y_test, rf_prediction)
    
    # SVM
    model_exists = False
    try:
        f = open("Pickled_models\\svm_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    if model_exists:
        pkl_filename = "Pickled_models\\svm_model.pkl"
        # Load from file
        with open(pkl_filename, 'rb') as file:
            svm_model = pickle.load(file)
            svm_model.fit(x_train, y_train)
            svm_prediction = svm_model.predict(x_test)
            svm_accuracy = accuracy_score(y_test, svm_prediction)
    else:
        svm_model = svm.SVC()
        svm_model.fit(x_train, y_train)
        svm_prediction = svm_model.predict(x_test)
        svm_accuracy = accuracy_score(y_test, svm_prediction)
    
    # KNN
    model_exists = False
    try:
        f = open("Pickled_models\\knn_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    if model_exists:
        pkl_filename = "Pickled_models\\knn_model.pkl"
        # Load from file
        with open(pkl_filename, 'rb') as file:
            knn_model = pickle.load(file)
            knn_model.fit(x_train, y_train)
            knn_prediction = knn_model.predict(x_test)
            knn_accuracy = accuracy_score(y_test, knn_prediction)
    else:
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_train, y_train)
        knn_prediction = knn_model.predict(x_test)
        knn_accuracy = accuracy_score(y_test, knn_prediction)
    
    #NN
    #* StandardScaler is ESSENTIAL for NN
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)   
    
    model_exists = False
    try:
        f = open("Pickled_models\\nn_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    if model_exists:
        pkl_filename = "Pickled_models\\nn_model.pkl"
        # Load from file
        with open(pkl_filename, 'rb') as file:
            nn_model = pickle.load(file)
            nn_model.fit(x_train, y_train)
            nn_prediction = nn_model.predict(x_test)
            nn_accuracy = accuracy_score(y_test, nn_prediction)
    else:
        nn_model = MLPClassifier(solver= 'adam', alpha=1e-5, hidden_layer_sizes=(200, 100),max_iter = 1000)
        nn_model.fit(x_train, y_train)
        nn_prediction = nn_model.predict(x_test)
        nn_accuracy = accuracy_score(y_test, nn_prediction)
         
    # Gauss
    try:
        f = open("Pickled_models\\nn_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    if model_exists:
        pkl_filename = "Pickled_models\\gauss_model.pkl"
        # Load from file
        with open(pkl_filename, 'rb') as file:
            gauss_model = pickle.load(file)
            gauss_model.fit(x_train, y_train)
            gauss_prediction = nn_model.predict(x_test)
            gauss_accuracy = accuracy_score(y_test, gauss_prediction)
    else:
        gauss_model = GaussianNB()
        gauss_model.fit(x_train, y_train)
        gauss_prediction = gauss_model.predict(x_test)
        gauss_accuracy = accuracy_score(y_test, gauss_prediction)
       
        
    rf_accuracy = (rf_accuracy)*100

    string +=("--------------------------------Training Accuracy RF--------------------------------\n")
    string +=(f'Accuracy: {rf_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {rf_prediction[:4]}\n')

    svm_accuracy = (svm_accuracy)*100
    
    string +=("--------------------------------Training Accuracy SVM--------------------------------\n")
    string +=(f'Accuracy: {svm_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {svm_prediction[:4]}\n')

    knn_accuracy = (knn_accuracy)*100

    string +=("--------------------------------Training Accuracy KNN--------------------------------\n")
    string +=(f'Accuracy: {knn_accuracy} %\n')

    nn_accuracy = (nn_accuracy)*100

    string +=("--------------------------------Training Accuracy Neural Network--------------------------------\n")
    string +=(f'Accuracy: {nn_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {nn_prediction[:4]}\n')
    
    gauss_accuracy = (gauss_accuracy)*100

    string +=("--------------------------------Training Accuracy Naive Bayes--------------------------------\n")
    string +=(f'Accuracy: {gauss_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {gauss_prediction[:4]}\n')


    with open("Pickled_models\\rf_model.pkl", 'wb') as file:
        pickle.dump(rf_model, file)

    with open("Pickled_models\\svm_model.pkl", 'wb') as file:
        pickle.dump(svm_model, file)
    
    with open("Pickled_models\\knn_model.pkl", 'wb') as file:
        pickle.dump(knn_model, file)
    
    with open("Pickled_models\\nn_model.pkl", 'wb') as file:
        pickle.dump(nn_model, file)
    
    with open("Pickled_models\\gauss_model.pkl", 'wb') as file:
        pickle.dump(gauss_model, file)

    return [rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy, gauss_accuracy,string]
                
def test_models(extract_path, dataIS, first_participant):
    string = ''
    
    print('Test')
    
    df = pd.DataFrame()
    if dataIS =='EEG':
        h = pd.read_csv(f"{extract_path}\\ERP_VP001_0-back.csv")
            
        df_0_back = pd.read_csv(f"{extract_path}\\ERP_VP0{participant}_0-back.csv")
        Y_0_back = [0]*df_0_back.shape[0]
        df_0_back.columns = h.columns
        df_0_back.insert(0,'Y',Y_0_back)
        
        df_2_back = pd.read_csv(f"{extract_path}\\ERP_VP0{participant}_2-back.csv")
        Y_2_back = [2]*df_2_back.shape[0]
        df_2_back.columns = h.columns
        df_2_back.insert(0,'Y',Y_2_back)
                        
        df_3_back = pd.read_csv(f"{extract_path}\\ERP_VP0{participant}_3-back.csv")
        Y_3_back = [3]*df_3_back.shape[0]
        df_3_back.columns = h.columns
        df_3_back.insert(0,'Y',Y_3_back)
        
        hold = df_0_back.copy()
        hold = hold.append(df_2_back, ignore_index = True)
        hold = hold.append(df_3_back, ignore_index = True)
        
        # Removes the NaN from EEG and NIRS
        if hold.isnull().values.any() and dataIS =='EEG':
            for i in range(hold.shape[0]):
                if hold.isna().any(axis=1)[i]:
                    hold  = hold[hold.index != i]
    
        X = hold.iloc[:,2:].to_numpy()
        Y = hold.iloc[:,0].to_numpy()

    elif dataIS == "NIRS":
        # r=root, d=directories, f = files
        for r, d, f in os.walk(extract_path):
            for file in f:
                if 'average_210315_1325.csv' not in file and 'All_Data.csv' not in file and '.csv' in file:
                    files.append(os.path.join(r, file))
            
        # print(files)
        h = pd.read_csv(files[0])
        if 'Unnamed' in h.columns[-1]:
            del h['Unnamed: 109']
            print(h.head())
        
        hold = pd.DataFrame()
        first_part = 0
        
        for i in range(len(files)):
            this_participant = f'{files[i][files[i].find("VP0")+3]}{files[i][files[i].find("VP0")+4]}'
            if this_participant == first_participant:
                first_part = i
                break
                
        
        for i in range(first_part,len(files)):
            this_participant = f'{files[i][files[i].find("VP0")+3]}{files[i][files[i].find("VP0")+4]}'
            if i+1 != len(files):
                next_participant = f'{files[i+1][files[i+1].find("VP0")+3]}{files[i+1][files[i+1].find("VP0")+4]}'
            elif i+1 == len(files):
                next_participant = ''
            else:
                print("We have an issue")
            
            #if this_participant == to next_participant then collect the data                 
            if this_participant == next_participant: 
                # print('The same')
                # print('This participant:', this_participant)
                # print('Next participant:', next_participant)
                if f'VP0{this_participant}' in files[i] and '0-back' in files[i]:
                    df_0_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_0_back.columns[-1]:
                        del df_0_back['Unnamed: 109']
                    
                    Y_0_back = [0]*df_0_back.shape[0]
                    df_0_back.columns = h.columns
                    df_0_back.insert(0,'Y',Y_0_back)
                    
                    hold = hold.append(df_0_back, ignore_index = True)
                    
                elif f'VP0{this_participant}' in files[i] and '2-back' in files[i]:
                    df_2_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_2_back.columns[-1]:
                        del df_2_back['Unnamed: 109']
                    
                    Y_2_back = [2]*df_2_back.shape[0]
                    df_2_back.columns = h.columns
                    df_2_back.insert(0,'Y',Y_2_back)
                    
                    hold = hold.append(df_2_back, ignore_index = True)
            
                elif f'VP0{this_participant}' in files[i] and '3-back' in files[i]:
                    df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back['Unnamed: 109']
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
            #if this_participant != to next_participant then analyze the data                 
            elif this_participant != next_participant:
                print('Not the same')
                print(hold.head())
                # print('This participant:', this_participant)
                # print('Next participant:', next_participant)
                X = hold.iloc[:,2:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_val = pickle_model_train(X,Y)
                
                #reseting hold
                hold = pd.DataFrame()

            #last_participant reached
            else:
                print('Last Participant:', this_participant)
                print(hold.head())
                X = hold.iloc[:,2:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_val = pickle_model_train(X,Y)

def pickle_model_test(X,Y):
    model_exists = False
    string = ''
    # We wont divide the data into training and testing sets, this tests the model with new data
    # x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
    x_test = X 
    y_test = Y
    
    # Load Random Forest Model
    pkl_filename = "Pickled_models\\rf_model.pkl"
    with open(pkl_filename, 'rb') as file:
        rf_model = pickle.load(file)
        rf_prediction = rf_model.predict(x_test)
        rf_accuracy = accuracy_score(y_test, rf_prediction)
    
    # Load SVM
    pkl_filename = "Pickled_models\\svm_model.pkl"
    with open(pkl_filename, 'rb') as file:
        svm_model = pickle.load(file)
        svm_prediction = svm_model.predict(x_test)
        svm_accuracy = accuracy_score(y_test, svm_prediction)
            
    # Load KNN
    pkl_filename = "Pickled_models\\knn_model.pkl"
    with open(pkl_filename, 'rb') as file:
        knn_model = pickle.load(file)
        knn_prediction = knn_model.predict(x_test)
        knn_accuracy = accuracy_score(y_test, knn_prediction)
            
    # Load NN
    pkl_filename = "Pickled_models\\nn_model.pkl"
    with open(pkl_filename, 'rb') as file:
        nn_model = pickle.load(file)
        nn_prediction = knn_model.predict(x_test)
        nn_accuracy = accuracy_score(y_test, nn_prediction)
        
            
    # Load Gauss
    pkl_filename = "Pickled_models\\gauss_model.pkl"
    with open(pkl_filename, 'rb') as file:
        gauss_model = pickle.load(file)
        gauss_prediction = gauss_model.predict(x_test)
        gauss_accuracy = accuracy_score(y_test, gauss_prediction)
    
    rf_accuracy = (rf_accuracy)*100

    string +=("--------------------------------Testing Accuracy RF--------------------------------\n")
    string +=(f'Accuracy: {rf_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {rf_prediction[:4]}\n')

    svm_accuracy = (svm_accuracy)*100
    
    string +=("--------------------------------Testing Accuracy SVM--------------------------------\n")
    string +=(f'Accuracy: {svm_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {svm_prediction[:4]}\n')

    knn_accuracy = (knn_accuracy)*100

    string +=("--------------------------------Testing Accuracy KNN--------------------------------\n")
    string +=(f'Accuracy: {knn_accuracy} %\n')

    nn_accuracy = (nn_accuracy)*100

    string +=("--------------------------------Testing Accuracy Neural Network--------------------------------\n")
    string +=(f'Accuracy: {nn_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {nn_prediction[:4]}\n')
    
    gauss_accuracy = (gauss_accuracy)*100

    string +=("--------------------------------Testing Accuracy Naive Bayes--------------------------------\n")
    string +=(f'Accuracy: {gauss_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {gauss_prediction[:4]}\n')


    with open("Pickled_models\\rf_model.pkl", 'wb') as file:
        pickle.dump(rf_model, file)

    with open("Pickled_models\\svm_model.pkl", 'wb') as file:
        pickle.dump(svm_model, file)
    
    with open("Pickled_models\\knn_model.pkl", 'wb') as file:
        pickle.dump(knn_model, file)
        
    with open("Pickled_models\\nn_model.pkl", 'wb') as file:
        pickle.dump(nn_model, file)
        
    with open("Pickled_models\\gauss_model.pkl", 'wb') as file:
        pickle.dump(gauss_model, file)

    
    

    return [rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy,gauss_accuracy,string]

def run_models(numOfIter, X, Y, select_col):
    
    rf_model_exists = False
    svm_model_exists = False
    knn_model_exists = False
    nn_model_exists = False
    gauss_model_exists = False
    
    rf_accuracy = 0
    svm_accuracy = 0
    knn_accuracy = 0
    nn_accuracy = 0
    gauss_accuracy = 0
    consensus_accuracy = 0
    string = ''
    
    for _ in range(numOfIter): 
        # Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)
        
        # Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=500,random_state = 0)
        rf_model.fit(x_train, y_train)
        rf_prediction = rf_model.predict(x_test)
        rf_accuracy += accuracy_score(y_test, rf_prediction)
        
        pkl_filename = "pickle_model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        
        # SVM
        svm_model = svm.SVC()
        svm_model.fit(x_train, y_train)
        svm_prediction = svm_model.predict(x_test)
        svm_accuracy += accuracy_score(y_test, svm_prediction)
        
        try:
            f = open("svm_model.pkl")
            rf_model_exists = True
            f.close()
        except FileNotFoundError:
            print("Could not open")
        
        
        # KNN
        # TODO Play around with the amount fo neighbors
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(x_train, y_train)
        knn_prediction = knn_model.predict(x_test)
        knn_accuracy += accuracy_score(y_test, knn_prediction)
        
        try:
            f = open("knn_model.pkl")
            rf_model_exists = True
            f.close()
        except FileNotFoundError:
            print("Could not open")
        

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
        
        try:
            f = open("nn_model.pkl")
            rf_model_exists = True
            f.close()
        except FileNotFoundError:
            print("Could not open")
        
        
        # Gauss
        # TODO Play around with the amount fo neighbors
        gauss_model = GaussianNB()
        gauss_model.fit(x_train, y_train)
        gauss_prediction = gauss_model.predict(x_test)
        gauss_accuracy += accuracy_score(y_test, gauss_prediction)
        
        try:
            f = open("gauss_model.pkl")
            rf_model_exists = True
            f.close()
        except FileNotFoundError:
            print("Could not open")
        
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
    
    # print(string)
    
    return [rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy, consensus_accuracy, string]

def cross_validate(numOfIter, X, Y, select_col):
    
    rf_accuracy = 0
    svm_accuracy = 0
    knn_accuracy = 0
    nn_accuracy = 0
    gauss_accuracy = 0
    consensus_accuracy = 0
    string = ''
    
    # Dividing the data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)
    
    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=500,random_state = 0)
    rf_accuracy = np.mean(cross_val_score(rf_model,X,Y,cv = numOfIter)) #numOfIter = 5
    
    # SVM
    svm_model = svm.SVC()
    svm_accuracy = np.mean(cross_val_score(svm_model,X,Y,cv = numOfIter))
    
    # KNN
    # TODO Play around with the amount fo neighbors
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_accuracy = np.mean(cross_val_score(knn_model,X,Y,cv = numOfIter))

    # NN with Keras
    #* define the keras model

    # #* StandardScaler is ESSENTIAL for NN
    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.fit_transform(x_test)
    
    # nn_model = Sequential()
    # nn_model.add(Dense(12, input_dim=8, activation='relu'))
    # nn_model.add(Dense(8, activation='relu'))
    # nn_model.add(Dense(1, activation='sigmoid'))
    
    # compile the keras model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # nn_accuracy = np.mean(cross_val_score(nn_model,X,Y,cv = numOfIter))
    # nn_model = make_pipeline(preprocessing.StandardScaler()
    #                          ,MLPClassifier(solver= 'lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10),max_iter = 1000))
    
    
    # Gauss
    # TODO Play around with the amount of neighbors
    gauss_model = GaussianNB()
    gauss_accuracy = np.mean(cross_val_score(gauss_model,X,Y,cv = numOfIter))
    
    # print([rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy])
    
    return [rf_accuracy, svm_accuracy, knn_accuracy, gauss_accuracy]

if __name__ == '__main__':
    # x is data type (EEG or NIRS), y is boolean (restrict data or not)
    # x = str(sys.argv[1])
    # y = str(sys.argv[2])
    # rows_names = []
    x = 'NIRS'
    y = ''
    z = 'False'
    a = 'True'
    
    if z == 'True':
        window_analysis = True
    elif z == 'False':
        window_analysis = False
        
    if a == 'True':
        tts_participant = True
    elif a == 'False':
        tts_participant = False
    
    print(y)
    
    if y == 'frontal':
        if x == "EEG":
            rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']
        elif x == "NIRS":
            rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']
    
    elif y == 'parietal':
        if x == "EEG":
            rows_names=  ['FC5','FC1','T7','C3','Cz','CP5','CP1','FC2','FC6','C4','T8','CP2','CP6','P4']
        elif x == "NIRS":
            rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']    
            
    elif y == 'occipital':
        if x == "EEG":
            rows_names=  ['P7','P3','Pz', 'P4','P8', 'POz', 'O1','O2']
        elif x == "NIRS":
            rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']
    else:
        rows_names = y
            
    
    all_data_file_exists = False
        
    sys.stdout.write(str(exec_Code(x, rows_names, window_analysis,tts_participant)))