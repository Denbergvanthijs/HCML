import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import tensorflow as tf
import data_loading


class Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_metrics(self, y_pred):
        y_test = self.y_test
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

    def create_model(self, loss_func='binary_crossentropy'):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train

        opt = keras.optimizers.Adam(learning_rate=0.0001)

        # Build the neural network model
        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.5))
        # model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))

        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss=loss_func, optimizer=opt, metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        return y_pred

# Exaplme of running the code

# X_train, X_test, y_train, y_test = data_loading.data_preparation()

# create and train a model, it will return the predictions made by the model in binary
# model = Model(X_train, X_test, y_train, y_test)
# pred = model.create_model()

# if you want to get the metrics from it
# model.get_metrics(pred)

