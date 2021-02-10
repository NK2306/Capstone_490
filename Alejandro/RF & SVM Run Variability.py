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

dataset = r'C:\Users\owner\Documents\GitHub\Capstone_490\Alejandro\TrainingData\Merged\merged.csv'
# Reading the information from the csv file into the dataset
datasets = pd.read_csv(dataset)
datasets.head()

# Prepearing the Data for training
#* Data must be divided into attributes and labels
X = datasets.iloc[:,1:5].values
Y = datasets.iloc[:,0].values

le = preprocessing.LabelEncoder()
le.fit(datasets['Source.Name'])
Y = le.fit_transform(Y)

X,Y = shuffle(X,Y)

count = 10

def random_forest():
    rf_accuracy = 0
    
    for _ in range(count): 
        #Dividing the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
        
        # Feature scaling
        #* We need to do this because our values in the dataset are in different scales, some in tens others in thousands
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        
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
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        
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
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        
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
    # print(f"X: {X[:5]}")
    # print(f"Y: {Y[:5]}")