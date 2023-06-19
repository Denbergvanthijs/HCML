from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def create_data_split():
    # Split the dataset into train and test sets
    data = pd.read_excel('data_taiwan/default of credit card clients.xls', skiprows=[1])
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Further split the train_data into train and validation sets
    test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

    # Save the train, test, and validation sets to CSV files
    train_data.to_csv('data_taiwan/train.csv', index=False)
    test_data.to_csv('data_taiwan/test.csv', index=False)
    val_data.to_csv('data_taiwan/validation.csv', index=False)

def data_preparation():
    #data = pd.read_excel('data_taiwan/default of credit card clients.xls', skiprows=[1])
    data_train = pd.read_csv('data_taiwan/train.csv')
    data_val = pd.read_csv('data_taiwan/validation.csv') 
    data_test = pd.read_csv('data_taiwan/test.csv')

    data_train['X2'] = data_train['X2']-1
    data_test['X2'] = data_test['X2']-1

    # create an x and y 
    X_train = data_train.drop(columns=['Y'],axis=1)
    X_test = data_test.drop(columns=['Y'],axis=1)

    y_train = data_train['Y']
    y_test = data_test['Y']

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled,X_test_scaled,y_train,y_test
