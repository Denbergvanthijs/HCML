from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import data_loading


class Baseline:
    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

    def create_model(self, loss_func="binary_crossentropy", learning_rate=0.0001, l2_reg=0.01):
        # Build the neural network model
        model = Sequential()
        model.add(Dense(50, activation="relu", input_dim=self.X_train.shape[1], kernel_regularizer=l2(l2_reg)))
        model.add(Dense(1, activation="sigmoid"))

        # Compile the model
        opt = Adam(learning_rate=learning_rate)
        model.compile(loss=loss_func, optimizer=opt, metrics=["accuracy"])

        self.model = model  # Save the model to the class

        return model

    def train_model(self, model=None, epochs=10, batch_size=32, verbose=1):
        if model is None:  # Check if no model is passed as arg
            if self.model is None:  # Check if no model is created yet
                print("No model passed, creating a new model")
                self.create_model()  # Create a model
            else:
                model = self.model  # Use the model saved in the class, if it exists
        # Else use the model passed as arg

        # Train the model
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                      validation_data=(self.X_val, self.y_val))

        return self.history

    @staticmethod
    def get_metrics(y_true, y_pred, threshold=0.5):
        y_pred_binary = (y_pred > threshold).astype(int)

        # Calculate different evaluation metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        confusion_mat = confusion_matrix(y_true, y_pred_binary)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "matrix": confusion_mat}


# Example of running the code
# Drop the intersection column from the data since we dont want it as a feature
fps = ["./data/intersectional_train.csv", "./data/intersectional_val.csv", "./data/intersectional_test.csv"]
X_train, X_val, X_test, y_train, y_val, y_test = data_loading.data_preparation(fps=fps, drop=["intersection"])

# Create and train a model, it will return the predictions made by the model in binary
baseline = Baseline(X_train, X_val, y_train, y_val)
model = baseline.create_model(loss_func="binary_crossentropy", learning_rate=0.0001, l2_reg=0.01)
history = baseline.train_model(model, epochs=10, batch_size=32, verbose=1)
print(history.history)

# Save model to disk
model.save("./models/baseline.h5")

# Predict the test set
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
pred_test = model.predict(X_test)

# If you want to get the metrics from it
metrics_train = Baseline.get_metrics(y_train, pred_train)
print(metrics_train)
metrics_val = Baseline.get_metrics(y_val, pred_val)
print(metrics_val)
metrics_test = Baseline.get_metrics(y_test, pred_test)
print(metrics_test)
