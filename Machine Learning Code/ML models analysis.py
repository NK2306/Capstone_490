import sys
import pandas as pd
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle 
from sklearn.model_selection import cross_val_score 
from sklearn.pipeline import make_pipeline
# from keras.models import Sequential
# from keras.layers import Activation,Dense
# from keras.metrics import categorical_crossentropy
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import math
from itertools import islice
import warnings
# warnings.filterwarnings("ignore")
import random

def exec_Code(data_path, dataIS, row_name, window_analysis, just_do, input_percent, remove_neg_time):
    #Constants
    rows_avg = 10
    count = 5
    files = []
    all_data_file_exists = False
    select_col = False
    window_size = 100
    last_train_participant = 21
    hold = pd.DataFrame()
            
    extract_path = f'{data_path}\\{dataIS}_Data'
    read_path = f'{data_path}\\{dataIS}_Data\\All_Data\\All_Data.csv'

    try:
        f = open(read_path)
        all_data_file_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    # Reading the information from the csv file into the dataset
    # if scrape_data:
    #     if not all_data_file_exists:
    #         scrape_files(extract_path, read_path, rows_avg, dataIS)
    #     else:
    #         print('The data file already exists')

    #* Implements a Just Train or Just Test option
    # walk & find how many particip there are
    for r, d, f in os.walk(extract_path):
        for file in f:
            if '.csv' in file and 'All_Data' not in file:
                files.append(os.path.join(r, file))
   
    # We assume that all the participants are there from 0 to len(files)
    last_participant = f'{files[-1][files[-1].find("VP0")+3]}{files[-1][files[-1].find("VP0")+4]}'
    
    if just_do != '':
        if just_do == 'Train':
            last_train_participant = last_participant
            acc_val = train_models(extract_path, dataIS, last_train_participant)
        
        if just_do == 'Test':
            first_test_participant = f'{files[0][files[0].find("VP0")+3]}{files[0][files[0].find("VP0")+4]}'
            acc_val = test_models(extract_path,dataIS,first_test_participant)
        return
    elif just_do == '':
        # Calc % put in by user
        if input_percent != '':
            rf_accuracy = 0
            svm_accuracy = 0
            knn_accuracy = 0
            nn_accuracy = 0
            gauss_accuracy = 0
            SGD_accuracy = 0
            consensus_accuracy = 0
            c=0
            
            #* Only take int the first 2 values & assume greater than 10?
            num_of_train = round((int(f'{input_percent[0]}{input_percent[1]}')/100.0)*(int(last_participant)))
            train_participant_list = []
            
            for _ in range(3):
                test_list = []
                #Generate n random numbers between 10 and 30
                print(f'This list {c}')
                train_participant_list = random.sample(range(1,int(last_participant)+1), num_of_train)
                # train_participant_list = random.sample(range(21,27), 3)
                print('Train list:',train_participant_list)
                
                for i in range(1,int(last_participant)+1):
                # for i in range(21,27):
                    if i not in train_participant_list:
                        test_list.append(i)
                print('Test list:', test_list)
                
                if row_name == '':
                    # Test the model  
                    train_models(extract_path, dataIS, train_participant_list)

                    # Test the model
                    acc_values = test_models(extract_path,dataIS,test_list)
                
                elif row_name != '':
                    # Test the model  
                    train_models(extract_path, dataIS, train_participant_list,row_name)

                    # Test the model
                    acc_values = test_models(extract_path,dataIS,test_list,row_name)
                
                rf_accuracy += acc_values[0]
                svm_accuracy += acc_values[1]
                knn_accuracy += acc_values[2]
                nn_accuracy += acc_values[3]
                gauss_accuracy += acc_values[4]
                SGD_accuracy += acc_values[5]
                consensus_accuracy += acc_values[6]
                c+= 1
                
                # Reset values
                os.remove("Pickled_models\\rf_model.pkl")
                os.remove("Pickled_models\\svm_model.pkl")
                os.remove("Pickled_models\\knn_model.pkl")
                os.remove("Pickled_models\\nn_model.pkl")
                os.remove("Pickled_models\\gauss_model.pkl")
                os.remove("Pickled_models\\SGD_model.pkl")
            rf_accuracy = rf_accuracy/c
            svm_accuracy = svm_accuracy/c
            knn_accuracy = knn_accuracy/c
            nn_accuracy = nn_accuracy/c
            gauss_accuracy = gauss_accuracy/c
            consensus_accuracy = consensus_accuracy/c
            SGD_accuracy = SGD_accuracy/c
            
            print("Data type:", dataIS)
            print("Overall Cross Validation Accuracy")
            print(show_accuracy(rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy, gauss_accuracy, consensus_accuracy, SGD_accuracy))
            
            accuracyFile = open("AccuracyResults.txt", "w")
            accuracyFile.write("Overall Cross Validation Accuracy")
            accuracyFile.write(show_accuracy(rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy, gauss_accuracy, consensus_accuracy, SGD_accuracy))
            accuracyFile.close()
            #return
        
        else:
            print('No Percentage Given, Use Default')
            print("Train test split inter-participants")
            print("Data type:", dataIS)
            
            last_train_participant = round((80)/100.0*(int(last_participant)))
            
            if row_name == '':
                # Test the model  
                train_models(extract_path, dataIS, last_train_participant)

                # Test the model
                acc_values = test_models(extract_path,dataIS,last_train_participant)
            elif row_name != '':
                # Test the model  
                train_models(extract_path, dataIS, last_train_participant,row_name)

                # Test the model
                acc_values = test_models(extract_path,dataIS,last_train_participant,row_name)
            
            #! Remove later
            # Reset values
            # os.remove("Pickled_models\\rf_model.pkl")
            # os.remove("Pickled_models\\svm_model.pkl")
            # os.remove("Pickled_models\\knn_model.pkl")
            # os.remove("Pickled_models\\nn_model.pkl")
            # os.remove("Pickled_models\\gauss_model.pkl")
            # os.remove("Pickled_models\\SGD_model.pkl")
            

        
    #* Sliding window analysis to get accuracy in that time frame
    if window_analysis:
        print("Sliding window analysis")
        window(window_size, extract_path, last_train_participant, dataIS, row_name)
                       
    # else:
    #     print(f"Data type: {dataIS}")
    #     print(f"Restrict Columns: {select_col}\n")
            
    #     # Removes the NaN from NIRS
    #     if np.isnan(hold.iloc[2,hold.shape[1]-1]) and dataIS =='NIRS':
    #         del hold[f'{hold.shape[1]-2}']
            
    #     print("Accuracy analysis with cross-validation")

    #     X = hold.iloc[:,1:].to_numpy()
    #     Y = hold.iloc[:,0].to_numpy()
    #     X,Y = shuffle(X,Y)
    #     # print(all_data_single_file(count,X, Y, select_col))
    
def check_columns(extract_path, dataframe, row_name, dataIS,):
    
    # h = pd.read_csv(read_path)
    hold = pd.DataFrame()
    hold.insert(0,'Y',dataframe['Y'])
    
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
                        hold.insert(len(hold.columns),col,dataframe[f'{df.columns[c]}'])
            c+=1
            
    elif dataIS == "EEG":
        # df = pd.read_csv(f'{extract_path}\\ERP_VP001_0-back.csv')
        for col in dataframe.columns:
            # print(col)
            # print(c)
            for r in range(len(row_name)):
                # print(row_name[r])
                if row_name[r] == col:
                    hold.insert(len(hold.columns),col,dataframe[f'{col}']) 
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
                
    if 'Unnamed' in hold.columns[-1]:
        del hold[f'{hold.columns[-1]}']
        # print(hold.head())
    
    hold.to_csv(read_path, index=False)

def window(window_size, extract_path, last_participant,dataIS,row_name):
    print("window")
    skip_num = int(window_size/10)
    df = pd.DataFrame()
    time_frame = []
    files = []
    rf_accuracy = []
    svm_accuracy = []
    knn_accuracy = []
    nn_accuracy = []
    gauss_accuracy = []
    SGD_accuracy = []
    consensus_accuracy = []
    c=0
    
    for r, d, f in os.walk(extract_path):
        for file in f:
            if '.csv' in file and 'All_Data' not in file:
                files.append(os.path.join(r, file))
                break
    h = pd.read_csv(files[0])
    
    if dataIS == "EEG":
        # time_frame.append('0.0')

        print("Untested, Errors expected")
        
        for j in range(0,int(h.shape[0]-skip_num),skip_num):  #int(h.shape[0]-skip_num)
            # df_0_back = pd.read_csv(f"{extract_path}\\VP001-NIRS\\0-back_session\\data_0-back_session_average_210315_1325.csv", skiprows = j+skip_num, nrows = window_size)
            # df_0_back.columns = h.columns
            # time_frame.append("{:.3f}".format(np.mean(df_0_back['Time'])))
            for i in range(-10,10):
                time_frame.append(i/10.0)
            
            # train phase
            train_models(extract_path,dataIS,last_participant,window_size,j,row_name)
                    
            # test phase    
            acc_values = test_models(extract_path,dataIS,last_participant,window_size,j,row_name)
            
            if c==0:
                rf_accuracy.append(acc_values[0])
                svm_accuracy.append(acc_values[1])
                knn_accuracy.append(acc_values[2])
                nn_accuracy.append(acc_values[3])
                gauss_accuracy.append(acc_values[4])
                SGD_accuracy.append(acc_values[5])
                consensus_accuracy.append(acc_values[6])
                c+=1
            rf_accuracy.append(acc_values[0])
            svm_accuracy.append(acc_values[1])
            knn_accuracy.append(acc_values[2])
            nn_accuracy.append(acc_values[3])
            gauss_accuracy.append(acc_values[4])
            SGD_accuracy.append(acc_values[5])
            consensus_accuracy.append(acc_values[6])
            
            # for i in range(len(time_frame)):
            #     if float(time_frame[i])>10.0:
            #         break
            
            #reset values
            os.remove("Pickled_models\\rf_model.pkl")
            os.remove("Pickled_models\\svm_model.pkl")
            os.remove("Pickled_models\\knn_model.pkl")
            os.remove("Pickled_models\\nn_model.pkl")
            os.remove("Pickled_models\\gauss_model.pkl")
            acc_values = 0
            
        for i in range(0,len(rf_accuracy)):
            time_frame.append(i)
        
        df.insert(0,'Random Forest',rf_accuracy)
        df.insert(1,'SVM',svm_accuracy)
        df.insert(2,'KNN',knn_accuracy)
        df.insert(3,'NN',nn_accuracy)
        df.insert(4,'Gauss',gauss_accuracy)
        df.insert(5,'SGD',SGD_accuracy)
        df.insert(6,'Consensus',consensus_accuracy)
        df.insert(7,'Time', time_frame)
        lines = df.plot(kind = 'line',x='Time',xlim=(-1,1),ylim=(0,1),xticks=(10),yticks=([w*0.1 for w in range(10)]))
        lines.set_title(f'Sliding Window Performance NIRS All Trials & Time Mean')
        lines.set_ylabel('Accuracy %')
        plt.legend(loc="best")
        plt.plot(df['Time'],consensus_accuracy)
        plt.savefig(f'Sliding Window Performance NIRS All Trials & Time Mean.png')
    
    elif dataIS == "NIRS":
        for j in range(0,int(h.shape[0]-skip_num),skip_num):  #int(h.shape[0]-skip_num)
            c+=1
            # train phase
            train_models(extract_path,dataIS,last_participant,row_name,window_size,j)
                    
            # test phase    
            acc_values = test_models(extract_path,dataIS,last_participant,row_name,window_size,j)

            rf_accuracy.append(acc_values[0])
            svm_accuracy.append(acc_values[1])
            knn_accuracy.append(acc_values[2])
            nn_accuracy.append(acc_values[3])
            gauss_accuracy.append(acc_values[4])
            SGD_accuracy.append(acc_values[5])
            consensus_accuracy.append(acc_values[6])
            
            #reset values
            # print('reset values')
            os.remove("Pickled_models\\rf_model.pkl")
            os.remove("Pickled_models\\svm_model.pkl")
            os.remove("Pickled_models\\knn_model.pkl")
            os.remove("Pickled_models\\nn_model.pkl")
            os.remove("Pickled_models\\gauss_model.pkl")
            os.remove("Pickled_models\\SGD_model.pkl")
            acc_values = 0

        for i in range(0,len(rf_accuracy)):
            time_frame.append(i)
        
        df.insert(0,'Random Forest',rf_accuracy)
        df.insert(1,'SVM',svm_accuracy)
        df.insert(2,'KNN',knn_accuracy)
        df.insert(3,'NN',nn_accuracy)
        df.insert(4,'Gauss',gauss_accuracy)
        df.insert(5,'SGD',SGD_accuracy)
        df.insert(6,'Consensus',consensus_accuracy)
        df.insert(7,'Time', time_frame)
        lines = df.plot(kind = 'line',x='Time',xlim=(0,60),ylim=(0,1),xticks=([w*5 for w in range(60)]),yticks=([w*0.1 for w in range(10)]))
        lines.set_title(f'Sliding Window Performance NIRS All Trials & Time Mean')
        lines.set_ylabel('Accuracy %')
        plt.legend(loc="best")
        # print(time_frame)
        # for t in range(len(time_frame)):
        #     if -0.1 <= float(time_frame[t]) <= 0.1:
        #         plt.axvline(x=t)
        #         break
        # plt.show()
        plt.savefig(f'Sliding Window Performance NIRS All Trials & Time Mean.png')   

def train_models(extract_path,dataIS, last_participant,row_name = '',window_size=0,jump=-1):
    print('train')
    c=0
    numOfIter = 100
    skip_num = int(window_size/10)
    # skip_num = 5
    
    # print(type(last_participant))
    if type(last_participant) == list:
        print('Is a list')
        
        if dataIS == "EEG":
            files = []
            i = 0
            # r=root, d=directories, f = files
            for r, d, f in os.walk(extract_path):
                for file in f:
                    if 'average' not in file and 'All_Data.csv' not in file and '.csv' in file:
                    # if 'average' in file and '.csv' in file:
                        files.append(os.path.join(r, file))
            
            h = pd.read_csv(files[2])
            if 'Unnamed' in h.columns[-1]:
                del h[f'{h.columns[-1]}'] 
            h = h.T
            if type(h.columns[-1]) == int:
                if type(h.iloc[0][0]) == str:
                    h.columns = h.iloc[0]
                    h = h.drop(index = f'{h.index[0]}')
            # print(h.head())
            
            hold = pd.DataFrame()
            fit_once_df = pd.DataFrame()
            last_particip = last_participant[-1]
            if int(last_particip) <10:
                    last_particip = '0'+f'{last_particip}'
            
            for this_participant in last_participant:
                if this_participant <10:
                    this_participant = '0'+f'{this_participant}'
                print('This participant is',this_participant)
                
                for i in range(len(files)-1):
                    if  'trial001' in files[i]:
                        continue
                    
                    if f'VP0{this_participant}' in files[i]:
                        # print('file: ',files[i])
                        if '0back' in files[i]:
                            # print('0-back')
                            if window_size != 0 and jump != -1:
                                df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_0_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_0_back.columns[-1]:
                                del df_0_back[f'{df_0_back.columns[-1]}']
                                
                            df_0_back = df_0_back.T
                            
                            if type(df_0_back.columns[-1]) == int:
                                if type(df_0_back.iloc[0][0]) == str:
                                    df_0_back.columns = h.columns
                                    df_0_back = df_0_back.drop(index = f'{df_0_back.index[0]}')
                            # print(df_0_back.head())
                            
                            # Removes the NaN from EEG
                            if df_0_back.isnull().values.any() and dataIS =='EEG':
                                for i in range(df_0_back.shape[0]):
                                    if df_0_back.isna().any(axis=1)[i]:
                                        df_0_back  = df_0_back[df_0_back.index != i]
                            
                            if window_size != 0 and jump != -1:
                                df_0_back.columns = h.columns
                                df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_0_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_0_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_0_back = df2
                            
                            Y_0_back = [0]*df_0_back.shape[0]
                            df_0_back.insert(0,'Y',Y_0_back)
                            
                            hold = hold.append(df_0_back, ignore_index = True)
                        
                        elif '2back' in files[i]:
                            # print('2-back')
                            if window_size != 0 and jump != -1:
                                df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                                
                            else:
                                df_2_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_2_back.columns[-1]:
                                del df_2_back[f'{df_2_back.columns[-1]}']
                                
                            df_2_back = df_2_back.T
                            if type(df_2_back.columns[-1]) == int:
                                if type(df_2_back.iloc[0][0]) == str:
                                    df_2_back.columns = h.columns
                                    df_2_back = df_2_back.drop(index = f'{df_2_back.index[0]}')
                    
                            # Removes the NaN from EEG
                            if df_2_back.isnull().values.any() and dataIS =='EEG':
                                for i in range(df_2_back.shape[0]):
                                    if df_2_back.isna().any(axis=1)[i]:
                                        df_2_back  = df_2_back[df_2_back.index != i]
                            
                            if window_size != 0 and jump != -1:
                                df_2_back.columns = h.columns
                                df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_2_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_2_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_2_back = df2
                            
                            Y_2_back = [2]*df_2_back.shape[0]
                            df_2_back.insert(0,'Y',Y_2_back)
                            
                            hold = hold.append(df_2_back, ignore_index = True)
                        
                        elif '3back' in files[i]:
                            if window_size != 0 and jump != -1:
                                df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_3_back = pd.read_csv(files[i])
                            
                            if 'Unnamed' in df_3_back.columns[-1]:
                                del df_3_back[f'{df_3_back.columns[-1]}']
                            
                            df_3_back = df_3_back.T
                            if type(df_3_back.columns[-1]) == int:
                                if type(df_3_back.iloc[0][0]) == str:
                                    df_3_back.columns = h.columns
                                    df_3_back = df_3_back.drop(index = f'{df_3_back.index[0]}')
                                    
                            # Removes the NaN from EEG
                            if df_3_back.isnull().values.any() and dataIS =='EEG':
                                for i in range(df_3_back.shape[0]):
                                    if df_3_back.isna().any(axis=1)[i]:
                                        df_3_back  = df_3_back[df_3_back.index != i]
                            
                            if window_size != 0 and jump != -1:
                                df_3_back.columns = h.columns
                                df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_3_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_3_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_3_back = df2
                            
                            Y_3_back = [3]*df_3_back.shape[0]
                            df_3_back.columns = h.columns
                            df_3_back.insert(0,'Y',Y_3_back)
                            
                            hold = hold.append(df_3_back, ignore_index = True)
                        
                        # print('here',i , len(files))
                        
                        if f'VP0{this_participant}' not in files[i+1]:
                            # print('Run train_models')
                            
                            # print(hold)
                            
                            if row_name != '':
                                hold = check_columns(extract_path, hold, row_name, dataIS)
                                # print(hold)
                            
                            X = hold.iloc[:,1:].to_numpy()
                            Y = hold.iloc[:,0].to_numpy()
                            acc_val = train_model_partial_fit(X,Y)
                            
                            # Build partial_fit dataframe
                            fit_once_df = fit_once_df.append(hold, ignore_index = True)
                            
                            #reseting hold
                            hold = pd.DataFrame()

                if f'{this_participant}' == f'{last_particip}':
                    # print('Run partial fit')
                    print('Last participant is',last_particip)
                    print(fit_once_df)
                    # # For last participant partial fit remaining models
                    X = fit_once_df.iloc[:,1:].to_numpy()
                    Y = fit_once_df.iloc[:,0].to_numpy()
                    train_model_fit_once(X,Y)
        
        if dataIS == "NIRS":
            files = []
            i = 0
            # r=root, d=directories, f = files
            for r, d, f in os.walk(extract_path):
                for file in f:
                    if 'average' not in file and 'All_Data.csv' not in file and '.csv' in file:
                    # if 'average' in file and '.csv' in file:
                        files.append(os.path.join(r, file))
            
            h = pd.read_csv(files[0])
            if 'Unnamed' in h.columns[-1]:
                del h[f'{h.columns[-1]}']
            
            hold = pd.DataFrame()
            fit_once_df = pd.DataFrame()
            last_particip = last_participant[-1]
            if int(last_particip) <10:
                    last_particip = '0'+f'{last_particip}'
            
            for this_participant in last_participant:
                if this_participant <10:
                    this_participant = '0'+f'{this_participant}'
                print('This participant is',this_participant)
                
                for i in range(len(files)-1):
                    # print (i)
                    if f'VP0{this_participant}' in files[i]:
                        # print('file: ',files[i])
                        if '0-back' in files[i]:
                            # print('0-back')
                            if window_size != 0 and jump != -1:
                                df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_0_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_0_back.columns[-1]:
                                del df_0_back[f'{df_0_back.columns[-1]}']
                            
                            df_0_back.columns = h.columns
                            
                            if window_size != 0 and jump != -1:
                                df_0_back.columns = h.columns
                                df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_0_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_0_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_0_back = df2
                                                
                            Y_0_back = [0]*df_0_back.shape[0]
                            df_0_back.insert(0,'Y',Y_0_back)
                            
                            hold = hold.append(df_0_back, ignore_index = True)
                        elif '2-back' in files[i]:
                            # print('2-back')
                            if window_size != 0 and jump != -1:
                                df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                                
                            else:
                                df_2_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_2_back.columns[-1]:
                                del df_2_back[f'{df_2_back.columns[-1]}']
                            
                            df_2_back.columns = h.columns
                            
                            if window_size != 0 and jump != -1:
                                df_2_back.columns = h.columns
                                df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_2_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_2_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_2_back = df2
                            
                            Y_2_back = [2]*df_2_back.shape[0]
                            df_2_back.insert(0,'Y',Y_2_back)
                            
                            hold = hold.append(df_2_back, ignore_index = True)
                        elif '3-back' in files[i]:
                            if window_size != 0 and jump != -1:
                                df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_3_back = pd.read_csv(files[i])
                            
                            if 'Unnamed' in df_3_back.columns[-1]:
                                del df_3_back[f'{df_3_back.columns[-1]}']
                            
                            if window_size != 0 and jump != -1:
                                df_3_back.columns = h.columns
                                df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_3_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_3_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_3_back = df2
                            
                            Y_3_back = [3]*df_3_back.shape[0]
                            df_3_back.columns = h.columns
                            df_3_back.insert(0,'Y',Y_3_back)
                            
                            hold = hold.append(df_3_back, ignore_index = True)
                        
                        # print('here',i , len(files))
                        
                        if f'VP0{this_participant}' not in files[i+1]:
                            # print('Run train_models')
                            
                            if row_name != '':
                                hold = check_columns(extract_path,hold, row_name, dataIS)
                            
                            # print(hold)
                            
                            X = hold.iloc[:,1:].to_numpy()
                            Y = hold.iloc[:,0].to_numpy()
                            acc_val = train_model_partial_fit(X,Y)
                            
                            # Build partial_fit dataframe
                            fit_once_df = fit_once_df.append(hold, ignore_index = True)
                            
                            #reseting hold
                            hold = pd.DataFrame()

                if f'{this_participant}' == f'{last_particip}':
                    # print('Run partial fit')
                    print('Last participant is',last_particip)
                    
                    # # For last participant partial fit remaining models
                    X = fit_once_df.iloc[:,1:].to_numpy()
                    Y = fit_once_df.iloc[:,0].to_numpy()
                    train_model_fit_once(X,Y)
        return
    
    df = pd.DataFrame()
    if dataIS =='EEG':
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(extract_path):
            for file in f:
                if 'All_Data' not in file and '.csv' in file:
                # if 'average' in file and '.csv' in file:
                    files.append(os.path.join(r, file))
        
        h = pd.read_csv(files[2])
        if 'Unnamed' in h.columns[-1]:
            del h[f'{h.columns[-1]}'] 
        h = h.T
        if type(h.columns[-1]) == int:
            if type(h.iloc[0][0]) == str:
                h.columns = h.iloc[0]
                h = h.drop(index = f'{h.index[0]}')
        
        fit_once_df = pd.DataFrame()
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
                if  'trial001' in files[i]:
                    continue
                
                if f'VP0{this_participant}' in files[i] and '0back' in files[i]:
                    # print('0-back')
                    if window_size != 0 and jump != -1:
                        df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_0_back = pd.read_csv(files[i])

                    if 'Unnamed' in df_0_back.columns[-1]:
                        del df_0_back[f'{df_0_back.columns[-1]}'] 
                    
                    df_0_back = df_0_back.T
                            
                    if type(df_0_back.columns[-1]) == int:
                        if type(df_0_back.iloc[0][0]) == str:
                            df_0_back.columns = h.columns
                            df_0_back = df_0_back.drop(index = f'{df_0_back.index[0]}')
                    # print(df_0_back.head())
                    
                    # Removes the NaN from EEG
                    if df_0_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_0_back.shape[0]):
                            if df_0_back.isna().any(axis=1)[i]:
                                df_0_back  = df_0_back[df_0_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_0_back.columns = h.columns
                        df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_0_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_0_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_0_back = df2
                    
                    # print(df_0_back)
                    
                    Y_0_back = [0]*df_0_back.shape[0]
                    # df_0_back.columns = h.columns
                    df_0_back.insert(0,'Y',Y_0_back)
                    
                    hold = hold.append(df_0_back, ignore_index = True)
                    
                elif f'VP0{this_participant}' in files[i] and '2back' in files[i]:
                    # print('2-back')
                    if window_size != 0 and jump != -1:
                        df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                        
                    else:
                        df_2_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_2_back.columns[-1]:
                        del df_2_back[f'{df_2_back.columns[-1]}'] 
                    
                    df_2_back = df_2_back.T
                    if type(df_2_back.columns[-1]) == int:
                        if type(df_2_back.iloc[0][0]) == str:
                            df_2_back.columns = h.columns
                            df_2_back = df_2_back.drop(index = f'{df_2_back.index[0]}')
            
                    # Removes the NaN from EEG
                    if df_2_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_2_back.shape[0]):
                            if df_2_back.isna().any(axis=1)[i]:
                                df_2_back  = df_2_back[df_2_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_2_back.columns = h.columns
                        df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_2_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_2_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_2_back = df2
                    
                    Y_2_back = [2]*df_2_back.shape[0]
                    # df_2_back.columns = h.columns
                    df_2_back.insert(0,'Y',Y_2_back)
                    
                    hold = hold.append(df_2_back, ignore_index = True)
                
            #if this_participant != to next_participant then analyze the data
            elif next_participant != f'{last_participant}':
                if f'VP0{this_participant}' in files[i] and '3back' in files[i]:
                    # print("3-back")
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                        # print("here1")
                    else:
                        df_3_back = pd.read_csv(files[i])
                        # print('Why we here?')
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back[f'{df_3_back.columns[-1]}'] 
                    
                    df_3_back = df_3_back.T
                    if type(df_3_back.columns[-1]) == int:
                        if type(df_3_back.iloc[0][0]) == str:
                            df_3_back.columns = h.columns
                            df_3_back = df_3_back.drop(index = f'{df_3_back.index[0]}')
                            
                    # Removes the NaN from EEG
                    if df_3_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_3_back.shape[0]):
                            if df_3_back.isna().any(axis=1)[i]:
                                df_3_back  = df_3_back[df_3_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    # df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
                print('This participant:',this_participant)
                
                if row_name != '':
                    hold = check_columns(extract_path,hold, row_name, dataIS)
                
                # Removes the NaN from EEG 
                if hold.isnull().values.any() and dataIS =='EEG':
                    for i in range(hold.shape[0]):
                        if hold.isna().any(axis=1)[i]:
                            hold  = hold[hold.index != i]
                
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_val = train_model_partial_fit(X,Y)
                
                # Build partial_fit dataframe
                fit_once_df = fit_once_df.append(hold, ignore_index = True)
                
                #reseting hold
                hold = pd.DataFrame()
                
            #last_participant reached
            else:
                if f'VP0{this_participant}' in files[i] and '3back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back[f'{df_3_back.columns[-1]}'] 
                    
                    df_3_back = df_3_back.T
                    
                    if type(df_3_back.columns[-1]) == int:
                        if type(df_3_back.iloc[0][0]) == str:
                            df_3_back.columns = h.columns
                            df_3_back = df_3_back.drop(index = f'{df_3_back.index[0]}')
                            
                    # Removes the NaN from EEG
                    if df_3_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_3_back.shape[0]):
                            if df_3_back.isna().any(axis=1)[i]:
                                df_3_back  = df_3_back[df_3_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    # df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
                print('Last Participant:', this_participant)
                # print(hold.head())
                    
                if row_name != '':
                    hold = check_columns(extract_path,hold, row_name, dataIS)
                
                if hold.isnull().values.any() and dataIS =='EEG':
                    for i in range(hold.shape[0]):
                        if hold.isna().any(axis=1)[i]:
                            hold  = hold[hold.index != i]
                
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                train_model_partial_fit(X,Y)
                
                # For last participant partial fit remaining models
                fit_once_df = fit_once_df.append(hold, ignore_index = True)
                X = fit_once_df.iloc[:,1:].to_numpy()
                Y = fit_once_df.iloc[:,0].to_numpy()
                train_model_fit_once(X,Y)
                
                #reseting hold
                hold = pd.DataFrame()
                break
    
    elif dataIS == "NIRS":
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(extract_path):
            for file in f:
                if 'average' not in file and 'All_Data.csv' not in file and '.csv' in file:
                # if 'average' in file and '.csv' in file:
                    files.append(os.path.join(r, file))
        
        h = pd.read_csv(files[0])
        if 'Unnamed' in h.columns[-1]:
            del h[f'{h.columns[-1]}']
        
        hold = pd.DataFrame()
        fit_once_df = pd.DataFrame()
        
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
                if f'VP0{this_participant}' in files[i] and '0-back' in files[i]:
                    # print('0-back')
                    if window_size != 0 and jump != -1:
                        df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_0_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_0_back.columns[-1]:
                        del df_0_back[f'{df_0_back.columns[-1]}']
                    
                    if window_size != 0 and jump != -1:
                        df_0_back.columns = h.columns
                        df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_0_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_0_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_0_back = df2
                                        
                    Y_0_back = [0]*df_0_back.shape[0]
                    df_0_back.insert(0,'Y',Y_0_back)
                    
                    hold = hold.append(df_0_back, ignore_index = True)
                    
                elif f'VP0{this_participant}' in files[i] and '2-back' in files[i]:
                    # print('2-back')
                    if window_size != 0 and jump != -1:
                        df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                        
                    else:
                        df_2_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_2_back.columns[-1]:
                        del df_2_back[f'{df_2_back.columns[-1]}']
                    
                    if window_size != 0 and jump != -1:
                        df_2_back.columns = h.columns
                        df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_2_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_2_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_2_back = df2
                    
                    Y_2_back = [2]*df_2_back.shape[0]
                    df_2_back.insert(0,'Y',Y_2_back)
                    
                    hold = hold.append(df_2_back, ignore_index = True)
                
            #if this_participant != to next_participant then analyze the data
            elif next_participant != f'{last_participant}':
                if f'VP0{this_participant}' in files[i] and '3-back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back[f'{df_3_back.columns[-1]}']
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
                # if this_participant == '04':
                #     #reseting hold
                #     hold = pd.DataFrame()
                #     continue
                # else:
                print('This participant:',this_participant)
                # print(hold)
                
                if row_name != '':
                    hold = check_columns(extract_path,hold, row_name, dataIS)
                
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_val = train_model_partial_fit(X,Y)
                
                # Build partial_fit dataframe
                fit_once_df = fit_once_df.append(hold, ignore_index = True)
                
                #reseting hold
                hold = pd.DataFrame()
                
            #last_participant reached
            else:
                if f'VP0{this_participant}' in files[i] and '3-back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back[f'{df_3_back.columns[-1]}']
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    # df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
                print('Last Participant:', this_participant)
                # print(hold.head())
                    
                if row_name != '':
                        hold = check_columns(extract_path,hold, row_name, dataIS)
                
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_val = train_model_partial_fit(X,Y)
                
                # For last participant partial fit remaining models
                fit_once_df = fit_once_df.append(hold, ignore_index = True)
                X = fit_once_df.iloc[:,1:].to_numpy()
                Y = fit_once_df.iloc[:,0].to_numpy()
                train_model_fit_once(X,Y)
                
                #reseting hold
                hold = pd.DataFrame()
                break
             
def train_model_partial_fit(X,Y):  
    model_exists = False
    string = ''
    # Dividing the data into training and testing sets
    #* Training size is 90% since this is the training set of data, 10% is used for preliminary testing
    # x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
    x_train = X
    y_train = Y
    
    #Pre-preprocessing
    #* StandardScaler is ESSENTIAL for NN
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_train,y_train = shuffle(x_train,y_train)
        
    #NN
    
    # model_exists = False
    # if os.path.exists('saved_model\\my_model'):
    #     model_exists = True
    # keras
    #     nn_model= tf.keras.models.load_model('saved_model/my_model')
    #     nn_model.partial_fit(x_train, y_train, epochs=31, batch_size=10)
    # else:
    #     # NN with Keras
    #     #* define the keras model
    #     nn_model = Sequential()
    #     nn_model.add(Dense(12, input_dim=108, activation='relu'))
    #     nn_model.add(Dense(8, activation='relu'))
    #     nn_model.add(Dense(1, activation='sigmoid'))
        
    #     # compile the keras model
    #     nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    #     nn_model.partial_fit(x=x_train, y=y_train, epochs=29, batch_size=10)
    
    # sklearn    
    model_exists = False 
    try:
        f = open("Pickled_models\\nn_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    if model_exists:
        pkl_filename = "Pickled_models\\nn_model.pkl"
        with open(pkl_filename, 'rb') as file:
            nn_model = pickle.load(file)
            nn_model.partial_fit(x_train, y_train)
    else:
        # nn_model = MLPClassifier(solver= 'adam', alpha=1e-5, hidden_layer_sizes=(50, 75, 100),max_iter = 250,learning_rate='adaptive')
        nn_model = MLPClassifier(activation = 'relu', hidden_layer_sizes = (100,), learning_rate = 'adaptive', max_iter = 450, solver = 'adam')
        nn_model.partial_fit(x_train, y_train,classes=np.unique(y_train))
        # %%time
        # params = {'activation': ['relu', 'tanh', 'logistic', 'identity','softmax'],
        #         'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,),(12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)],
        #         'solver': ['adam', 'sgd', 'lbfgs'],
        #         'max_iter': [150,200,250,300,400,500],
        #         'learning_rate' : ['constant', 'adaptive', 'invscaling']
        #         }

        # nn_model = GridSearchCV(MLPClassifier(), param_grid=params, n_jobs=-1, cv=5, scoring='accuracy')
        # nn_model.partial_fit(x_train,y_train)
    
    # Gauss
    model_exists = False 
    try:
        f = open("Pickled_models\\gauss_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    if model_exists:
        pkl_filename = "Pickled_models\\gauss_model.pkl"
        # Load from file
        with open(pkl_filename, 'rb') as file:
            gauss_model = pickle.load(file)
            gauss_model.partial_fit(x_train, y_train)
    else:
        gauss_model = GaussianNB()
        gauss_model.partial_fit(x_train, y_train,classes=np.unique(y_train))
    
    # Stochastic Gradient Descent (SGD)
    model_exists = False 
    try:
        f = open("Pickled_models\\SGD_model.pkl")
        model_exists = True
        f.close()
    except FileNotFoundError:
        print("Could not open")
    
    if model_exists:
        pkl_filename = "Pickled_models\\SGD_model.pkl"
        # Load from file
        with open(pkl_filename, 'rb') as file:
            SGD_model = pickle.load(file)
            SGD_model.partial_fit(x_train, y_train)
    else:
        SGD_model = SGDClassifier(loss="hinge", penalty="l2")#, max_iter=5
        SGD_model.partial_fit(x_train, y_train,classes=np.unique(y_train))
    
    with open("Pickled_models\\nn_model.pkl", 'wb') as file:
        pickle.dump(nn_model, file)
    
    # nn_model.save('saved_model/my_model')
    
    with open("Pickled_models\\gauss_model.pkl", 'wb') as file:
        pickle.dump(gauss_model, file)
        
    with open("Pickled_models\\SGD_model.pkl", 'wb') as file:
        pickle.dump(SGD_model, file)

#* We use this function to work around the fact that partial_fit() does not work with these models.
#* This is a work around to train these models once with the data that exists, but retraining it will overwrite the previous model.
def train_model_fit_once(X,Y):
    x_train = X
    y_train = Y
    
    #Pre-preprocessing
    #* StandardScaler is ESSENTIAL for NN
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_train,y_train = shuffle(x_train,y_train)
    
    # Random Forest Model
    # rf_model = RandomForestClassifier(n_estimators=250, max_features='auto',min_samples_leaf=25,n_jobs=-1,oob_score = True)
    rf_model = RandomForestClassifier(max_features='sqrt', n_estimators=500,min_samples_leaf=25, n_jobs=-1, oob_score='False')
    rf_model.fit(x_train, y_train)
    
    # Parameter Optimization
    # params = {'max_features': ['auto', 'sqrt', 'log2', 'None'],
    #         'min_samples_leaf': list(range(1,50,5)),
    #         'oob_score': ['True', 'False'],
    #         'n_estimators': [150,200,250,300,400,500]}
    # rf_model = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid=params, n_jobs=-1,cv=5, scoring='accuracy')
    # rf_model.fit(x_train, y_train)
    
    # SVM
    # svm_model = svm.SVC(C=100, gamma=0.001, kernel='sigmoid')
    svm_model = svm.SVC(C=10, gamma=0.01)
    svm_model.fit(x_train, y_train)
    
    # Parameter Optimization
    # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    # svm_model = GridSearchCV(svm.SVC(),param_grid,refit=True, scoring = 'accuracy')
    # svm_model.fit(x_train, y_train)
    
    # KNN
    # knn_model = KNeighborsClassifier(leaf_size=3, n_neighbors=3)
    knn_model = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
    knn_model.fit(x_train, y_train)
    
    # List Hyperparameters to tune
    # leaf_size = list(range(1,50))
    # n_neighbors = list(range(1,19))
    # p=[1,2]
    # #convert to dictionary
    # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    # #Making model
    # knn_model = GridSearchCV(KNeighborsClassifier(), hyperparameters, cv=10, scoring = 'accuracy')
    # knn_model.fit(x_train, y_train)
    
    # nn_model = MLPClassifier(solver= 'adam', alpha=1e-5, hidden_layer_sizes=(50, 75, 100),max_iter = 250,learning_rate='adaptive')
    # nn_model = MLPClassifier(activation = 'relu', hidden_layer_sizes = (100,), learning_rate = 'adaptive', max_iter = 450, solver = 'adam')
    # nn_model.fit(x_train, y_train)
    
    # Parameter Optimization
    # params = {'activation': ['relu', 'tanh', 'logistic', 'identity','softmax'],
    #         'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,),(15,),(20,)],
    #         'solver': ['adam', 'sgd', 'lbfgs'],
    #         'max_iter': [150,300,450,600,1000],
    #         'learning_rate' : ['constant', 'adaptive', 'invscaling']
    #         }

    # nn_model = GridSearchCV(MLPClassifier(), param_grid=params, n_jobs=-1, cv=5, scoring='accuracy')
    # nn_model.fit(x_train, y_train)
    
    # gauss_model = GaussianNB()
    # gauss_model.fit(x_train, y_train)
    
    # SGD_model = SGDClassifier(loss="hinge", penalty="l2") #, max_iter=5?
    # params = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    #         'penalty': ['l2', 'l1', 'elasticnet'],
    #         'max_iter': [300,550,600,750,900,1050,1200,1350],
    #         # 'learning_rate': ['constant','optimal','invscaling','adaptive']
    #         }

    # SGD_model = GridSearchCV(SGDClassifier(), param_grid=params, n_jobs=-1, cv=5, scoring='accuracy')
    # SGD_model.fit(x_train, y_train)
    
    with open("Pickled_models\\rf_model.pkl", 'wb') as file:
        pickle.dump(rf_model, file)

    with open("Pickled_models\\svm_model.pkl", 'wb') as file:
        pickle.dump(svm_model, file)
    
    with open("Pickled_models\\knn_model.pkl", 'wb') as file:
        pickle.dump(knn_model, file)
    
    # with open("Pickled_models\\nn_model.pkl", 'wb') as file:
    #     pickle.dump(nn_model, file)
    
    # with open("Pickled_models\\gauss_model.pkl", 'wb') as file:
    #     pickle.dump(gauss_model, file)
        
    # with open("Pickled_models\\SGD_model.pkl", 'wb') as file:
    #     pickle.dump(SGD_model, file)

def test_models(extract_path, dataIS, first_participant,row_name = '',window_size=0,jump=-1):
    string = ''
    rf_accuracy = 0
    svm_accuracy = 0
    knn_accuracy = 0
    nn_accuracy = 0
    gauss_accuracy = 0
    SGD_accuracy = 0
    consensus_accuracy = 0
    c=0
    numOfIter = 100
    skip_num = int(window_size/10)
    # skip_num = 5
    
    print('Test')
    if type(first_participant) == list:
        print('Is a list')
        
        if dataIS == "EEG":
            files = []
            i = 0
            # r=root, d=directories, f = files
            for r, d, f in os.walk(extract_path):
                for file in f:
                    if 'average' not in file and 'All_Data.csv' not in file and '.csv' in file:
                    # if 'average' in file and '.csv' in file:
                        files.append(os.path.join(r, file))
            
            h = pd.read_csv(files[2])
            if 'Unnamed' in h.columns[-1]:
                del h[f'{h.columns[-1]}'] 
            h = h.T
            if type(h.columns[-1]) == int:
                if type(h.iloc[0][0]) == str:
                    h.columns = h.iloc[0]
                    h = h.drop(index = f'{h.index[0]}')
            # print(h.head())
            
            hold = pd.DataFrame()
            fit_once_df = pd.DataFrame()
            last_particip = first_participant[-1]
            if int(last_particip) <10:
                    last_particip = '0'+f'{last_particip}'
            
            for this_participant in first_participant:
                if this_participant <10:
                    this_participant = '0'+f'{this_participant}'
                print('This participant is',this_participant)
                
                for i in range(len(files)-1):
                    if  'trial001' in files[i]:
                        continue
                    
                    if f'VP0{this_participant}' in files[i]:
                        # print('file: ',files[i])
                        if '0back' in files[i]:
                            # print('0-back')
                            if window_size != 0 and jump != -1:
                                df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_0_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_0_back.columns[-1]:
                                del df_0_back[f'{df_0_back.columns[-1]}']
                                
                            df_0_back = df_0_back.T
                            if type(df_0_back.columns[-1]) == int:
                                if type(df_0_back.iloc[0][0]) == str:
                                    df_0_back.columns = h.columns
                                    df_0_back = df_0_back.drop(index = f'{df_0_back.index[0]}')
                            # print(df_0_back.head())
                            
                            # Removes the NaN from EEG
                            if df_0_back.isnull().values.any() and dataIS =='EEG':
                                for i in range(df_0_back.shape[0]):
                                    if df_0_back.isna().any(axis=1)[i]:
                                        df_0_back  = df_0_back[df_0_back.index != i]
                            
                            if window_size != 0 and jump != -1:
                                df_0_back.columns = h.columns
                                df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_0_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_0_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_0_back = df2
                            
                            Y_0_back = [0]*df_0_back.shape[0]
                            df_0_back.insert(0,'Y',Y_0_back)
                            
                            hold = hold.append(df_0_back, ignore_index = True)
                        
                        elif '2back' in files[i]:
                            # print('2-back')
                            if window_size != 0 and jump != -1:
                                df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                                
                            else:
                                df_2_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_2_back.columns[-1]:
                                del df_2_back[f'{df_2_back.columns[-1]}']
                                
                            df_2_back = df_2_back.T
                            if type(df_2_back.columns[-1]) == int:
                                if type(df_2_back.iloc[0][0]) == str:
                                    df_2_back.columns = h.columns
                                    df_2_back = df_2_back.drop(index = f'{df_2_back.index[0]}')
                    
                            # Removes the NaN from EEG
                            if df_2_back.isnull().values.any() and dataIS =='EEG':
                                for i in range(df_2_back.shape[0]):
                                    if df_2_back.isna().any(axis=1)[i]:
                                        df_2_back  = df_2_back[df_2_back.index != i]
                            
                            if window_size != 0 and jump != -1:
                                df_2_back.columns = h.columns
                                df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_2_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_2_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_2_back = df2
                            
                            Y_2_back = [2]*df_2_back.shape[0]
                            df_2_back.insert(0,'Y',Y_2_back)
                            
                            hold = hold.append(df_2_back, ignore_index = True)
                        
                        elif '3back' in files[i]:
                            if window_size != 0 and jump != -1:
                                df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_3_back = pd.read_csv(files[i])
                            
                            if 'Unnamed' in df_3_back.columns[-1]:
                                del df_3_back[f'{df_3_back.columns[-1]}']
                            
                            df_3_back = df_3_back.T
                            if type(df_3_back.columns[-1]) == int:
                                if type(df_3_back.iloc[0][0]) == str:
                                    df_3_back.columns = h.columns
                                    df_3_back = df_3_back.drop(index = f'{df_3_back.index[0]}')
                                    
                            # Removes the NaN from EEG
                            if df_3_back.isnull().values.any() and dataIS =='EEG':
                                for i in range(df_3_back.shape[0]):
                                    if df_3_back.isna().any(axis=1)[i]:
                                        df_3_back  = df_3_back[df_3_back.index != i]
                            
                            if window_size != 0 and jump != -1:
                                df_3_back.columns = h.columns
                                df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_3_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_3_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_3_back = df2
                            
                            Y_3_back = [3]*df_3_back.shape[0]
                            df_3_back.columns = h.columns
                            df_3_back.insert(0,'Y',Y_3_back)
                            
                            hold = hold.append(df_3_back, ignore_index = True)
                        
                        # print('here',i , len(files))
                        
                        if f'VP0{this_participant}' not in files[i+1]:
                            # print('Run train_models')
                            
                            print(hold)
                            
                            if row_name != '':
                                hold = check_columns(extract_path, hold, row_name, dataIS)
                                print(hold)
                            
                            X = hold.iloc[:,1:].to_numpy()
                            Y = hold.iloc[:,0].to_numpy()
                            acc_values = pickle_model_test(X,Y)
                            if window_size == 0 and jump == -1:
                                print(acc_values[-1])
                            rf_accuracy += acc_values[0]
                            svm_accuracy += acc_values[1]
                            knn_accuracy += acc_values[2]
                            nn_accuracy += acc_values[3]
                            gauss_accuracy += acc_values[4]
                            SGD_accuracy += acc_values[5]
                            consensus_accuracy += acc_values[6]
                            string+=acc_values[-1]
                            c+= 1
                            
                            #reseting hold
                            hold = pd.DataFrame()
 
        if dataIS == "NIRS":
            files = []
            i = 0
            # r=root, d=directories, f = files
            for r, d, f in os.walk(extract_path):
                for file in f:
                    if 'average' not in file and 'All_Data.csv' not in file and '.csv' in file:
                    # if 'average' in file and '.csv' in file:
                        files.append(os.path.join(r, file))
            
            h = pd.read_csv(files[0])
            if 'Unnamed' in h.columns[-1]:
                del h[f'{h.columns[-1]}']
            
            hold = pd.DataFrame()
            fit_once_df = pd.DataFrame()
            last_particip = first_participant[-1]
            if int(last_particip) <10:
                    last_particip = '0'+f'{last_particip}'
            
            for this_participant in first_participant:
                if this_participant <10:
                    this_participant = '0'+f'{this_participant}'
                print('This participant is',this_participant)
                
                for i in range(len(files)-1):
                    # print (i)
                    if f'VP0{this_participant}' in files[i]:
                        # print('file: ',files[i])
                        if '0-back' in files[i]:
                            # print('0-back')
                            if window_size != 0 and jump != -1:
                                df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_0_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_0_back.columns[-1]:
                                del df_0_back[f'{df_0_back.columns[-1]}']
                            
                            if window_size != 0 and jump != -1:
                                df_0_back.columns = h.columns
                                df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_0_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_0_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_0_back = df2
                                                
                            Y_0_back = [0]*df_0_back.shape[0]
                            df_0_back.insert(0,'Y',Y_0_back)
                            
                            hold = hold.append(df_0_back, ignore_index = True)
                        elif '2-back' in files[i]:
                            # print('2-back')
                            if window_size != 0 and jump != -1:
                                df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                                
                            else:
                                df_2_back = pd.read_csv(files[i])
                        
                            if 'Unnamed' in df_2_back.columns[-1]:
                                del df_2_back[f'{df_2_back.columns[-1]}']
                            
                            if window_size != 0 and jump != -1:
                                df_2_back.columns = h.columns
                                df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_2_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_2_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_2_back = df2
                            
                            Y_2_back = [2]*df_2_back.shape[0]
                            df_2_back.insert(0,'Y',Y_2_back)
                            
                            hold = hold.append(df_2_back, ignore_index = True)
                        elif '3-back' in files[i]:
                            if window_size != 0 and jump != -1:
                                df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                            else:
                                df_3_back = pd.read_csv(files[i])
                            
                            if 'Unnamed' in df_3_back.columns[-1]:
                                del df_3_back[f'{df_3_back.columns[-1]}']
                            
                            if window_size != 0 and jump != -1:
                                df_3_back.columns = h.columns
                                df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                            else:
                                count = 0
                                df = pd.DataFrame(columns = h.columns)
                                df2 = pd.DataFrame(columns = h.columns)
                                for j in range(df_3_back.shape[0]):
                                    if count%numOfIter !=0 or count ==0:
                                        df = df.append(df_3_back.iloc[j], ignore_index = True)
                                        count += 1
                                    else:
                                        # print('Bye')
                                        df = df.mean(axis = 0, skipna = True).to_frame().T
                                        df2 = df2.append(df, ignore_index = True)
                                        df = pd.DataFrame()
                                        count = 0
                                df_3_back = df2
                            
                            Y_3_back = [3]*df_3_back.shape[0]
                            df_3_back.columns = h.columns
                            df_3_back.insert(0,'Y',Y_3_back)
                            
                            hold = hold.append(df_3_back, ignore_index = True)
                        
                        # print('here',i , len(files))
                        if f'VP0{this_participant}' not in files[i+1]:
                            # print('Run test_models')
                            if row_name != '':
                                hold = check_columns(extract_path,hold, row_name, dataIS)
                            
                            X = hold.iloc[:,1:].to_numpy()
                            Y = hold.iloc[:,0].to_numpy()
                            acc_values = pickle_model_test(X,Y)
                            if window_size == 0 and jump == -1:
                                print(acc_values[-1])
                            rf_accuracy += acc_values[0]
                            svm_accuracy += acc_values[1]
                            knn_accuracy += acc_values[2]
                            nn_accuracy += acc_values[3]
                            gauss_accuracy += acc_values[4]
                            SGD_accuracy += acc_values[5]
                            consensus_accuracy += acc_values[6]
                            string+=acc_values[-1]
                            c+= 1
                            
                            #reseting hold
                            hold = pd.DataFrame()

        rf_accuracy = rf_accuracy/c
        svm_accuracy = svm_accuracy/c
        knn_accuracy = knn_accuracy/c
        nn_accuracy = nn_accuracy/c
        gauss_accuracy = gauss_accuracy/c
        consensus_accuracy = consensus_accuracy/c
        SGD_accuracy = SGD_accuracy/c
        
        print("Overall List Accuracy")
        print(show_accuracy(rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy, gauss_accuracy, consensus_accuracy, SGD_accuracy))
        
        return [rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy, SGD_accuracy, consensus_accuracy, string]
    
    df = pd.DataFrame()
    if dataIS =='EEG':
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(extract_path):
            for file in f:
                if 'All_Data.csv' not in file and '.csv' in file:
                # if 'average' in file and '.csv' in file:
                    files.append(os.path.join(r, file))
        
        h = pd.read_csv(files[2])
        if 'Unnamed' in h.columns[-1]:
            del h[f'{h.columns[-1]}'] 
        h = h.T
        if type(h.columns[-1]) == int:
            if type(h.iloc[0][0]) == str:
                h.columns = h.iloc[0]
                h = h.drop(index = f'{h.index[0]}')
        # print(h.head())
        
        hold = pd.DataFrame()
        first_part = 0
        
        for i in range(len(files)):
            this_participant = f'{files[i][files[i].find("VP0")+3]}{files[i][files[i].find("VP0")+4]}'
            if this_participant == f'{first_participant}':
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
                if  'trial001' in files[i]:
                    continue
                
                if f'VP0{this_participant}' in files[i] and '0back' in files[i]:
                    # print('0-back')
                    if window_size != 0 and jump != -1:
                        df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_0_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_0_back.columns[-1]:
                        del df_0_back[f'{df_0_back.columns[-1]}'] 
                    
                    df_0_back = df_0_back.T
                    if type(df_0_back.columns[-1]) == int:
                        if type(df_0_back.iloc[0][0]) == str:
                            df_0_back.columns = h.columns
                            df_0_back = df_0_back.drop(index = f'{df_0_back.index[0]}')
                    # print(df_0_back.head())
                    
                    # Removes the NaN from EEG
                    if df_0_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_0_back.shape[0]):
                            if df_0_back.isna().any(axis=1)[i]:
                                df_0_back  = df_0_back[df_0_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_0_back.columns = h.columns
                        df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_0_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_0_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_0_back = df2
                    
                    Y_0_back = [0]*df_0_back.shape[0]
                    df_0_back.insert(0,'Y',Y_0_back)
                    
                    hold = hold.append(df_0_back, ignore_index = True)
                    
                elif f'VP0{this_participant}' in files[i] and '2back' in files[i]:
                    # print('2-back')
                    if window_size != 0 and jump != -1:
                        df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                        
                    else:
                        df_2_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_2_back.columns[-1]:
                        del df_2_back[f'{df_2_back.columns[-1]}'] 
                    
                    df_2_back = df_2_back.T
                    if type(df_2_back.columns[-1]) == int:
                        if type(df_2_back.iloc[0][0]) == str:
                            df_2_back.columns = h.columns
                            df_2_back = df_2_back.drop(index = f'{df_2_back.index[0]}')
                    # print(df_0_back.head())
                    
                    # Removes the NaN from EEG
                    if df_2_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_2_back.shape[0]):
                            if df_2_back.isna().any(axis=1)[i]:
                                df_2_back  = df_2_back[df_2_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_2_back.columns = h.columns
                        df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_2_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_2_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_2_back = df2
                    
                    Y_2_back = [2]*df_2_back.shape[0]
                    # df_2_back.columns = h.columns
                    df_2_back.insert(0,'Y',Y_2_back)
                    
                    hold = hold.append(df_2_back, ignore_index = True)
                
            #if this_participant != to next_participant then analyze the data
            elif next_participant != '':
                if  'trial001' in files[i]:
                    continue
                
                if f'VP0{this_participant}' in files[i] and '3back' in files[i]:
                    # print("3-back")
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                        # print("here1")
                    else:
                        df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back[f'{df_3_back.columns[-1]}'] 
                    
                    df_3_back = df_3_back.T
                    if type(df_3_back.columns[-1]) == int:
                        if type(df_3_back.iloc[0][0]) == str:
                            df_3_back.columns = h.columns
                            df_3_back = df_3_back.drop(index = f'{df_3_back.index[0]}')
                    # print(df_0_back.head())
                    
                    # Removes the NaN from EEG
                    if df_3_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_3_back.shape[0]):
                            if df_3_back.isna().any(axis=1)[i]:
                                df_3_back  = df_3_back[df_3_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    # df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)

                print('This participant:',this_participant)
                
                if row_name != '':
                    hold = check_columns(extract_path,hold, row_name, dataIS)
                    
                if hold.isnull().values.any() and dataIS =='EEG':
                    for i in range(hold.shape[0]):
                        if hold.isna().any(axis=1)[i]:
                            hold  = hold[hold.index != i]
                
                print(hold)
                
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_values = pickle_model_test(X,Y)
                if window_size == 0 and jump == -1:
                    print(acc_values[-1])
                rf_accuracy += acc_values[0]
                svm_accuracy += acc_values[1]
                knn_accuracy += acc_values[2]
                nn_accuracy += acc_values[3]
                gauss_accuracy += acc_values[4]
                SGD_accuracy += acc_values[5]
                consensus_accuracy += acc_values[6]
                string+=acc_values[-1]

                c+= 1
                
                #reseting hold
                hold = pd.DataFrame()
                
            #last_participant reached
            else:
                if  'trial001' in files[i]:
                    continue
                
                if f'VP0{this_participant}' in files[i] and '3back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back[f'{df_3_back.columns[-1]}'] 
                    
                    df_3_back = df_3_back.T
                    if type(df_3_back.columns[-1]) == int:
                        if type(df_3_back.iloc[0][0]) == str:
                            df_3_back.columns = h.columns
                            df_3_back = df_3_back.drop(index = f'{df_3_back.index[0]}')
                    # print(df_0_back.head())
                    
                    # Removes the NaN from EEG
                    if df_3_back.isnull().values.any() and dataIS =='EEG':
                        for i in range(df_3_back.shape[0]):
                            if df_3_back.isna().any(axis=1)[i]:
                                df_3_back  = df_3_back[df_3_back.index != i]
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for j in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[j], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    # df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
                print('Last Participant:', this_participant)
                # print(hold.head())
                    
                if row_name != '':
                        hold = check_columns(extract_path,hold, row_name, dataIS)
                
                if hold.isnull().values.any() and dataIS =='EEG':
                    for i in range(hold.shape[0]):
                        if hold.isna().any(axis=1)[i]:
                            hold  = hold[hold.index != i]
                
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_values = pickle_model_test(X,Y)
                if window_size == 0 and jump == -1:
                    print(acc_values[-1])
                rf_accuracy += acc_values[0]
                svm_accuracy += acc_values[1]
                knn_accuracy += acc_values[2]
                nn_accuracy += acc_values[3]
                gauss_accuracy += acc_values[4]
                SGD_accuracy += acc_values[5]
                consensus_accuracy += acc_values[6]
                string+=acc_values[-1]

                print(hold)
                
                c+= 1
                #reseting hold
                hold = pd.DataFrame()
                break

    elif dataIS == "NIRS":
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(extract_path):
            for file in f:
                if 'average_210315_1325.csv' not in file and 'All_Data.csv' not in file and '.csv' in file:
                # if 'average_210315_1325.csv' in file:
                    files.append(os.path.join(r, file))
            
        # print(files)
        h = pd.read_csv(files[0])
        if 'Unnamed' in h.columns[-1]:
            del h[f'{h.columns[-1]}']
            # print(h.head())
        
        hold = pd.DataFrame()
        first_part = 0
        
        for i in range(len(files)):
            this_participant = f'{files[i][files[i].find("VP0")+3]}{files[i][files[i].find("VP0")+4]}'
            if this_participant == f'{first_participant}':
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
                if f'VP0{this_participant}' in files[i] and '0-back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_0_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_0_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_0_back.columns[-1]:
                        del df_0_back[f'{df_0_back.columns[-1]}']
                    
                    
                    if window_size != 0 and jump != -1:
                        df_0_back.columns = h.columns
                        df_0_back = df_0_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_0_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_0_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_0_back = df2
                        
                    Y_0_back = [0]*df_0_back.shape[0]
                    # df_0_back.columns = h.columns
                    df_0_back.insert(0,'Y',Y_0_back)
                    
                    hold = hold.append(df_0_back, ignore_index = True)
                    
                elif f'VP0{this_participant}' in files[i] and '2-back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_2_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                        
                    else:
                        df_2_back = pd.read_csv(files[i])
                
                    if 'Unnamed' in df_2_back.columns[-1]:
                        del df_2_back[f'{df_2_back.columns[-1]}']

                    if window_size != 0 and jump != -1:
                        df_2_back.columns = h.columns
                        df_2_back = df_2_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_2_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_2_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_2_back = df2
                    
                    Y_2_back = [2]*df_2_back.shape[0]
                    # df_2_back.columns = h.columns
                    df_2_back.insert(0,'Y',Y_2_back)
                    
                    hold = hold.append(df_2_back, ignore_index = True)
            
            #if this_participant != to next_participant then analyze the data                 
            elif next_participant != '':
                print('This participant:', this_participant)
                if f'VP0{this_participant}' in files[i] and '3-back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_3_back = pd.read_csv(files[i])
                    
                    if 'Unnamed' in df_3_back.columns[-1]:
                        del df_3_back[f'{df_3_back.columns[-1]}']
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    # df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                
                if row_name != '':
                        hold = check_columns(extract_path,hold, row_name, dataIS)
                # print(hold.head())
                
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_values = pickle_model_test(X,Y)
                if window_size == 0 and jump == -1:
                    print(acc_values[-1])
                rf_accuracy += acc_values[0]
                svm_accuracy += acc_values[1]
                knn_accuracy += acc_values[2]
                nn_accuracy += acc_values[3]
                gauss_accuracy += acc_values[4]
                SGD_accuracy += acc_values[5]
                consensus_accuracy += acc_values[6]
                string+=acc_values[-1]
                c+= 1
                
                #reseting hold
                hold = pd.DataFrame()


            #last_participant reached
            else:
                if f'VP0{this_participant}' in files[i] and '3-back' in files[i]:
                    if window_size != 0 and jump != -1:
                        df_3_back = pd.read_csv(files[i], skiprows = jump+skip_num, nrows = window_size)
                    else:
                        df_3_back = pd.read_csv(files[i])
                    
                    if window_size != 0 and jump != -1:
                        df_3_back.columns = h.columns
                        df_3_back = df_3_back.mean(axis = 0, skipna = True).to_frame().T
                    else:
                        count = 0
                        df = pd.DataFrame(columns = h.columns)
                        df2 = pd.DataFrame(columns = h.columns)
                        for i in range(df_3_back.shape[0]):
                            if count%numOfIter !=0 or count ==0:
                                df = df.append(df_3_back.iloc[i], ignore_index = True)
                                count += 1
                            else:
                                # print('Bye')
                                df = df.mean(axis = 0, skipna = True).to_frame().T
                                df2 = df2.append(df, ignore_index = True)
                                df = pd.DataFrame()
                                count = 0
                        df_3_back = df2
                    
                    Y_3_back = [3]*df_3_back.shape[0]
                    # df_3_back.columns = h.columns
                    df_3_back.insert(0,'Y',Y_3_back)
                    
                    hold = hold.append(df_3_back, ignore_index = True)
                print('Last Participant:', this_participant)
                if row_name != '':
                        hold = check_columns(extract_path,hold, row_name, dataIS)
                # print(hold.head())
                    
                X = hold.iloc[:,1:].to_numpy()
                Y = hold.iloc[:,0].to_numpy()
                acc_values = pickle_model_test(X,Y)
                if window_size == 0 and jump == -1:
                    print(acc_values[-1])
                rf_accuracy += acc_values[0]
                svm_accuracy += acc_values[1]
                knn_accuracy += acc_values[2]
                nn_accuracy += acc_values[3]
                gauss_accuracy += acc_values[4]
                SGD_accuracy += acc_values[5]
                consensus_accuracy += acc_values[6]
                string+=acc_values[-1]
                c+= 1
                
                #reseting hold
                hold = pd.DataFrame()

    rf_accuracy = rf_accuracy/c
    svm_accuracy = svm_accuracy/c
    knn_accuracy = knn_accuracy/c
    nn_accuracy = nn_accuracy/c
    gauss_accuracy = gauss_accuracy/c
    consensus_accuracy = consensus_accuracy/c
    SGD_accuracy = SGD_accuracy/c
    
    print(dataIS)
    print("Overall Accuracy")
    print(show_accuracy(rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy, gauss_accuracy, consensus_accuracy, SGD_accuracy))
    
    return [rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy,SGD_accuracy, consensus_accuracy, string]

def pickle_model_test(X,Y):
    model_exists = False
    string = ''
    rf_accuracy = 0
    svm_accuracy = 0
    knn_accuracy = 0
    nn_accuracy = 0
    gauss_accuracy = 0
    consensus_accuracy = 0
    # We wont divide the data into training and testing sets, this tests the model with new data
    # x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)
    x_test = np.array(X)
    y_test = np.array(Y)
    
    # print('X:',X)
    # print('Y:',Y)
    # print(X.shape)
    # print(Y.shape)
    #* StandardScaler is ESSENTIAL for NN
    sc = StandardScaler()
    x_test = sc.fit_transform(x_test)
    
    # Load Random Forest Model
    pkl_filename = "Pickled_models\\rf_model.pkl"
    with open(pkl_filename, 'rb') as file:
        rf_model = pickle.load(file)
        rf_prediction = rf_model.predict(x_test)
        rf_accuracy = accuracy_score(y_test, rf_prediction)
        # print('RF Best Accuracy  :',rf_accuracy)
        # print('Best Parameters : ',rf_model.best_params_)
        # print('Best Estimators: ',rf_model.best_estimator_)
    
    # Load SVM
    pkl_filename = "Pickled_models\\svm_model.pkl"
    with open(pkl_filename, 'rb') as file:
        svm_model = pickle.load(file)
        svm_prediction = svm_model.predict(x_test)
        svm_accuracy = accuracy_score(y_test, svm_prediction)
        # print('SVM Best Accuracy  :',svm_accuracy)
        # print('Best Parameters : ',svm_model.best_params_)
        # print('Best Estimators: ',svm_model.best_estimator_)
            
    # Load KNN
    pkl_filename = "Pickled_models\\knn_model.pkl"
    with open(pkl_filename, 'rb') as file:
        knn_model = pickle.load(file)
        knn_prediction = knn_model.predict(x_test)
        knn_accuracy = accuracy_score(y_test, knn_prediction)
        # print('KNN Best Accuracy  :',knn_accuracy)
        # print('Best Parameters : ',knn_model.best_params_)
        # print('Best Estimators: ',knn_model.best_estimator_)
    
    #NN
    # #* StandardScaler is ESSENTIAL for NN
    # sc = StandardScaler()
    # x_test = sc.fit_transform(x_test)
    
    # Load NN
    # sklearn
    pkl_filename = "Pickled_models\\nn_model.pkl"
    with open(pkl_filename, 'rb') as file:
        nn_model = pickle.load(file)
        nn_prediction = nn_model.predict(x_test)
        nn_accuracy = accuracy_score(y_test, nn_prediction)
        # print('NN Best Accuracy  :',nn_accuracy)
        # print('Best Parameters : ',nn_model.best_params_)
        # print('Best Estimators: ',nn_model.best_estimator_)

    # Keras
    # nn_model= tf.keras.models.load_model('saved_model/my_model')
    # nn_accuracy = nn_model.predict(x_test, batch_size=10, verbose=False)
    # _, nn_accuracy = nn_model.evaluate(x_test, y_test)
    
    # Load Gauss
    pkl_filename = "Pickled_models\\gauss_model.pkl"
    with open(pkl_filename, 'rb') as file:
        gauss_model = pickle.load(file)
        gauss_prediction = gauss_model.predict(x_test)
        gauss_accuracy = accuracy_score(y_test, gauss_prediction)
        
    # Load SGD
    pkl_filename = "Pickled_models\\SGD_model.pkl"
    with open(pkl_filename, 'rb') as file:
        SGD_model = pickle.load(file)
        SGD_prediction = SGD_model.predict(x_test)
        SGD_accuracy = accuracy_score(y_test, SGD_prediction)
    #     print('SGD Best Accuracy  :',SGD_model)
    #     print('Best Parameters : ',SGD_model.best_params_)
    #     print('Best Estimators: ',SGD_model.best_estimator_)
    
        
    #* Check highest consensus of two models as a whole
    # Percentage similarity of lists 
    # Initialize to Neural Network
    consensus_prediction = nn_prediction
    
    # #* Checks consensus individually
    for i in range(len(nn_prediction)):
        # print("Checking consensus", i)
        if nn_prediction[i] == gauss_prediction[i]:
            consensus_prediction[i] = nn_prediction[i]
        
        # Consensus between SVM and Random Forest    
        elif rf_prediction[i] == svm_prediction[i]:
            consensus_prediction[i] = rf_prediction[i]
        
        # Consensus between Neural Network and KNN
        elif nn_prediction[i] == knn_prediction[i]:
            consensus_prediction[i] = nn_prediction[i]
            
        # Consensus between Neural Network and Random Forest
        elif nn_prediction[i] == rf_prediction[i]:
            consensus_prediction[i] = nn_prediction[i]
        
        # elif nn_prediction[i] == SGD_prediction[i]:
        #     consensus_prediction[i] = nn_prediction[i]
        
        # elif SGD_prediction[i] == gauss_prediction[i]:
        #     consensus_prediction[i] = nn_prediction[i]

        elif gauss_prediction[i] == rf_prediction[i]:
            consensus_prediction[i] = gauss_prediction[i]
        
        # Consensus between KNN and Random Forest
        elif gauss_prediction[i] == knn_prediction[i]:
            consensus_prediction[i] = rf_prediction[i]
        
        elif gauss_prediction[i] == svm_prediction[i]:
            consensus_prediction[i]= nn_prediction[i]
        
        # Consensus between Neural Network and SVM    
        elif nn_prediction[i] == svm_prediction[i]:
            consensus_prediction[i]= nn_prediction[i]
            
        
        # Consensus between KNN and SVM
        elif knn_prediction[i] == svm_prediction[i]:
            consensus_prediction[i] = knn_prediction[i]
            
        # Else default to NN
        else:
            consensus_prediction[i] = nn_prediction[i]
                
        consensus_accuracy = accuracy_score(y_test,consensus_prediction)
    
    string = show_accuracy(rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy, consensus_accuracy, SGD_accuracy)

    with open("Pickled_models\\rf_model.pkl", 'wb') as file:
        pickle.dump(rf_model, file)

    with open("Pickled_models\\svm_model.pkl", 'wb') as file:
        pickle.dump(svm_model, file)
    
    with open("Pickled_models\\knn_model.pkl", 'wb') as file:
        pickle.dump(knn_model, file)
        
    with open("Pickled_models\\nn_model.pkl", 'wb') as file:
        pickle.dump(nn_model, file)
    
    # nn_model.save('saved_model/my_model')
    
    with open("Pickled_models\\gauss_model.pkl", 'wb') as file:
        pickle.dump(gauss_model, file)  
    
    with open("Pickled_models\\SGD_model.pkl", 'wb') as file:
        pickle.dump(SGD_model, file)

    return [rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy,gauss_accuracy, SGD_accuracy, consensus_accuracy, string]

def show_accuracy(rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy, gauss_accuracy, consensus_accuracy, SGD_accuracy, string = ''):
    
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
    
    SGD_accuracy = (SGD_accuracy)*100
    string +=("--------------------------------Evaluating SGD Accuracy--------------------------------\n")
    string +=(f'Accuracy: {SGD_accuracy} %\n')
    
    consensus_accuracy = (consensus_accuracy)*100
    string +=("--------------------------------Evaluating Consensus Accuracy--------------------------------\n")
    string +=(f'Accuracy: {consensus_accuracy} %\n')
    
    return string

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
            
        # nn_model = MLPClassifier(solver= solver_string, alpha=1e-5, hidden_layer_sizes=(200, 100),max_iter = 1000)
        # nn_model.fit(x_train, y_train)
        # nn_prediction = nn_model.predict(x_test)
        # nn_accuracy += accuracy_score(y_test, nn_prediction)
        params = {'activation': ['relu', 'tanh', 'logistic', 'identity','softmax'],
          'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,)],
          'solver': ['adam', 'sgd', 'lbfgs'],
          'learning_rate' : ['constant', 'adaptive', 'invscaling']
         }

        mlp_clf_grid = GridSearchCV(MLPClassifier(), param_grid=params, n_jobs=-1, cv=5)
        mlp_clf_grid.fit(x_train,y_train)
        
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
        
    #     consensus_prediction = nn_model.predict(x_test)
        
    #     #* Check highest consensus of two models as a whole
    #     # Percentage similarity of lists 
        
    #     # #* Checks consensus individually
    #     for i in range(len(nn_prediction)):
    #         # Consensus between Neural Network and Random Forest
    #         if nn_prediction[i] == rf_prediction[i]:
    #             consensus_prediction[i] = nn_prediction[i]
                
    #         # Consensus between Neural Network and KNN
    #         elif nn_prediction[i] == knn_prediction[i]:
    #             consensus_prediction[i] = nn_prediction[i]
                
    #         # Consensus between KNN and Random Forest
    #         elif rf_prediction[i] == knn_prediction[i]:
    #             consensus_prediction[i] = rf_prediction[i]
                
    #         # Consensus between Neural Network and SVM    
    #         elif nn_prediction[i] == svm_prediction[i]:
    #             consensus_prediction[i]= nn_prediction[i]
                
    #         # Consensus between SVM and Random Forest    
    #         elif rf_prediction[i] == svm_prediction[i]:
    #             consensus_prediction[i] = rf_prediction[i]
                
    #         # Consensus between KNN and SVM
    #         elif knn_prediction[i] == svm_prediction[i]:
    #             consensus_prediction[i] = knn_prediction[i]
                
    #         # Else default to NN
    #         else:
    #             consensus_prediction[i] = nn_prediction[i]
                
                
                
    #     consensus_accuracy += accuracy_score(y_test,consensus_prediction)

    # string = show_accuracy(rf_accuracy, svm_accuracy, knn_accuracy,nn_accuracy, gauss_accuracy)
        
    # consensus_accuracy = (consensus_accuracy/numOfIter)*100
    # string +=("--------------------------------Evaluating Consensus Accuracy--------------------------------\n")
    # string +=(f'Accuracy: {consensus_accuracy} %\n')
    # string +=(f'y_test: {y_test[:4]}\nPrediction: {consensus_prediction[:4]}\n')
    
    # print(string)
    
    return [rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy, consensus_accuracy, string]

def all_data_single_file(numOfIter, X, Y, select_col):
    
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

    #Pre-preprocessing
    #* StandardScaler is ESSENTIAL for NN
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X,Y = shuffle(X,Y)
    
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.fit_transform(x_test)
    nn_model = Sequential([
    Dense(units = 16, input_dim=(108,), activation='relu'),
    Dense(units = 32, activation='relu'),
    Dense(units = 2, activation='softmax')])
    
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn_model.fit(x=X, y=Y, epochs=30, batch_size=10, validation_split=0.1)
    # nn_model = make_pipeline(preprocessing.StandardScaler()
    #                          ,MLPClassifier(solver= 'lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10),max_iter = 1000))
    nn_accuracy = np.mean(cross_val_score(nn_model,X,Y,cv = numOfIter))
    
    # Gauss
    # TODO Play around with the amount of neighbors
    gauss_model = GaussianNB()
    gauss_accuracy = np.mean(cross_val_score(gauss_model,X,Y,cv = numOfIter))
    
    # print([rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy, gauss_accuracy])
    
    return [rf_accuracy, svm_accuracy, knn_accuracy,gauss_accuracy]

def remove_negative_time(dataIS,files):
    # print(files)
    if dataIS =='EEG':
        df = pd.read_csv(files)
        
        if 'Unnamed' in df.columns[-1]:
            del df[f'{df.columns[-1]}']
        
        # Assuming data is transposed
        df = df.T
        if type(df.columns[0]) == int:
                if type(df.iloc[0][0]) == str:
                    df.columns = df.iloc[0]
                    df = df.drop(index = f'{df.index[0]}')
                    if 'Time' not in df.columns:
                        df.insert(0,'Time',df.index)
                    df.index = list(range(df.shape[0]))
        
        for i in range(df.shape[0]):
            if float(df['Time'][i]) <0:
                # print(df['Time'][i])
                df = df.drop([i], axis=0)
        df = df.T
        df.rename_axis(None, inplace=True)
        df.insert(0,'',df.index)
        df.columns = df.iloc[0]
        df = df.drop(index = f'{df.index[0]}')
        df.to_csv(files, index=False)
        
    if dataIS == "NIRS":
        df = pd.read_csv(files)
        for i in range(df.shape[0]):
            if df['Time'][i] <0:
                df = df.drop([i], axis=0)
        
        if 'Unnamed' in df.columns[-1]:
            del df[f'{df.columns[-1]}']
                    
        df.to_csv(files, index=False)

if __name__ == '__main__':
    # x is data type (EEG or NIRS), y is boolean (restrict data or not)
    # x = str(sys.argv[1])
    # y = str(sys.argv[2])
    # z = str(sys.argv[3])
    # a = str(sys.argv[4])
    # path = str(sys.argv[5])
    # rows_names = [] 
    path = 'C:\\Users\\owner\\Documents\\GitHub\\Capstone_490\\Alejandro'
    x = 'EEG'
    y = 'frontal'
    
    z = 'False'
    window_analysis = False
    if z == 'True':
        window_analysis = True
    
    just_do = ''
    remove_neg_time = False
    input_percent = ''

    print(y)
    if y == 'frontal':
        if x == "EEG":
            rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']
        elif x == "NIRS":
            print('This feature has not been implemented')
            # return
            # rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']
    
    elif y == 'parietal':
        if x == "EEG":
            rows_names=  ['FC5','FC1','T7','C3','Cz','CP5','CP1','FC2','FC6','C4','T8','CP2','CP6','P4']
        elif x == "NIRS":
            print('This feature has not been implemented')
            # return
            # rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']    
            
    elif y == 'occipital':
        if x == "EEG":
            rows_names=  ['P7','P3','Pz', 'P4','P8', 'POz', 'O1','O2']
        elif x == "NIRS":
            print('This feature has not been implemented')
            # return
            # rows_names=  ['FP1','FP2','F1','F2', 'AFz', 'AFF5','AFF6']
    else:
        rows_names = y
    
    all_data_file_exists = False
    if not remove_neg_time:
        sys.stdout.write(str(exec_Code(path, x, rows_names, window_analysis, just_do, input_percent, remove_neg_time)))
    
    # Remove negative time
    # r=root, d=directories, f = files
    elif remove_neg_time:
        for r, d, f in os.walk(extract_path):
            for file in f:
                if '.csv' in file and 'All_Data' not in file:
                    files.append(os.path.join(r, file))
        
        for i in range(len(files)):
            if dataIS == 'EEG' and 'trial001' in files[i]:
                continue
            remove_negative_time(dataIS,files[i])
            
        print('Files have been edited and negative time removed')
        
        sys.stdout.write(str(exec_Code(path, x, rows_names, window_analysis, just_do, input_percent, remove_neg_time)))
        
        print('Goodbye Friend')
    
