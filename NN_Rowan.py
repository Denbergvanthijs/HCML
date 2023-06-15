import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import numpy as np

import tensorflow as tf

def weighted_loss(y_true, y_pred, weights):
    # Define the standard loss function
    loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    # Multiply the loss by the adjusted weights
    weighted_loss = tf.reduce_mean(loss * weights)

    return weighted_loss

def categorise(row, weights):  
    """function for creating a new column with weights"""
    if row['X2'] == 0 and row['Y'] == 1:
        return weights[0]
    elif row['X2'] == 1 and row['Y'] == 1:
        return weights[1]
    elif row['X2'] == 0  and row['Y'] == 0:
        return weights[2]
    else:
        return weights[3]
    
#reweighing function for gender and defaulting
def Reweighing(protected_attribute, classifictation_class, dataset):

    default_list = np.bincount(classifictation_class) / len(classifictation_class)
    P_gender = np.bincount(protected_attribute) / len(protected_attribute)
    # Calculate P expected

    P_default = default_list[1]
    P_no_default = default_list[0]
    P_exp_default = P_gender*P_default
    P_exp_nodefault =P_gender*P_no_default
    P_exp_values = np.concatenate((P_exp_default, P_exp_nodefault))

    # Calculate P observed
    counts_observed = dataset.value_counts(['X2', 'Y'], normalize=True)
    P_observed = np.flip(counts_observed.values)
    weights = P_exp_values/P_observed #weights are default-male, default-female, no default male, no default female
    dataset['weights'] = dataset.apply(lambda row: categorise(row, weights), axis=1)

    return dataset

def get_metrics(y_pred, y_test):
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate different evaluation metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:')
    print(confusion_mat)

# load data
data = pd.read_excel('data_taiwan/default of credit card clients.xls', skiprows=[1])
data['X2'] = data['X2']-1

# create an x and y 
X = data.drop('Y', axis=1)
y = data['Y']
weighted_data = Reweighing(y, data['Y'], data)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
opt = keras.optimizers.Adam(learning_rate=0.0001)

# Build the neural network model
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.5))
# model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the original model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)
y_pred = model.predict(X_test_scaled)

get_metrics(y_pred, y_test)

# train model with reweighted loss function
weighted_data = Reweighing(y, data['Y'], data)
X = weighted_data.drop('Y', axis=1)
y = weighted_data['Y']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the same model with new loss functions
model.compile(loss=lambda y_true, y_pred: weighted_loss(y_true, y_pred, weighted_data['weights']), optimizer=opt, metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)
y_pred = model.predict(X_test_scaled)

get_metrics(y_pred, y_test)