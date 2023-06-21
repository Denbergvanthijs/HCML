import numpy as np
import tensorflow as tf
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from tensorflow.keras.models import load_model

from data_loading import data_preparation
from evaluation import plot_history
#from evaluation import UU_PAL
from NN import Baseline

# Load data from disk
fps = ["./data/intersectional_train.csv", "./data/intersectional_val.csv", "./data/intersectional_test.csv"]
X_train, X_val, X_test, y_train, y_val, y_test = data_preparation(fps=fps)


def weighted_loss(y_true, y_pred, weights):
    # Define the standard loss function
    loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    # Multiply the loss by the adjusted weights
    weighted_loss = tf.reduce_mean(loss * weights)

    return weighted_loss

def categorise(row, weights):  
    """function for creating a new column with weights"""
    if row['intersection'] == 11 and row['y'] == 1:
        return weights[0]
    elif row['intersection'] == 12 and row['y'] == 1:
        return weights[1]
    elif row['intersection'] == 21  and row['y'] == 1:
        return weights[2]
    elif row['intersection'] == 22  and row['y'] == 1:
        return weights[3]
    elif row['intersection'] == 11  and row['y'] == 0:
        return weights[4]
    elif row['intersection'] == 12  and row['y'] == 0:
        return weights[5]
    elif row['intersection'] == 21  and row['y'] == 0:
        return weights[6]
    else:
        return weights[7]
    
#reweighing function for gender and defaulting
def Reweighing_data(protected_attribute, classification_class, dataset):

    default_list = classification_class.value_counts(normalize=True)
    P_gender = protected_attribute.value_counts(normalize=True)

    # Calculate P expected

    P_default = default_list.values[1]
    P_no_default = default_list.values[0]
    P_exp_default = P_gender.values*P_default
    P_exp_nodefault =P_gender.values*P_no_default
    P_exp_values = np.concatenate((P_exp_default, P_exp_nodefault))

    # Calculate P observed
    counts_observed = dataset.value_counts(['intersection', 'y'], normalize=True)
    P_observed = np.flip(counts_observed.values)

    weights = P_exp_values/P_observed #weights are default-male married, default-male single, default-female married, default female single, no default-male married, no default-male single, no default-female married, no default female single
    dataset['weights'] = dataset.apply(lambda row: categorise(row, weights), axis=1)

    return dataset   

# create binary dataset with priv and unpriv groups
unpriv_group = [{"intersection": 11}, {"intersection": 12}]
priv_group = [{"intersection": 21}, {"intersection": 22}]

X_train_new = X_train.copy(deep=True)
X_train_new["target_column"] = y_train

X_test_new = X_test.copy(deep=True)
X_test_new["target_column"] = y_test

X_train_new = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=X_train_new,
                               label_names=["target_column"], protected_attribute_names=["intersection"])
X_test_new = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=X_test_new,
                              label_names=["target_column"], protected_attribute_names=["intersection"])



# apply reweighing
dataset_train1 = Reweighing(unprivileged_groups=unpriv_group,privileged_groups=priv_group).fit_transform(X_train_new)
weights = dataset_train1.instance_weights

# create a new model with the weighted loss function
reweighed = Baseline(X_train, X_val, y_train, y_val)
model = reweighed.create_model(loss_func=lambda y_true, y_pred: weighted_loss(y_true, y_pred, weights), learning_rate=0.0001, l2_reg=0.01)

history = reweighed.train_model(model, epochs=10, batch_size=32, verbose=1)

# Plot history
plot_history(history)

# Predict the test set
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
pred_test = model.predict(X_test)

# print metrics
metrics_train = Baseline.get_metrics(y_train, pred_train)
print('accuracy:', metrics_train['accuracy'], 'f1_score: ', metrics_train['f1'],   ' matrix: ', metrics_train['matrix'])
metrics_val = Baseline.get_metrics(y_val, pred_val)
print('accuracy:', metrics_val['accuracy'], 'f1_score: ', metrics_val['f1'],  ' matrix: ', metrics_val['matrix'])
metrics_test = Baseline.get_metrics(y_test, pred_test)
print('accuracy:', metrics_test['accuracy'],'f1_score: ', metrics_test['f1'],  ' matrix: ', metrics_test['matrix'])

# statistical parity before and after
metric_orig_train = BinaryLabelDatasetMetric(X_train_new,
                                             unprivileged_groups=unpriv_group,
                                             privileged_groups=priv_group)
print(("#### Original training dataset"))
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
      metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(X_test_new,
                                             unprivileged_groups=unpriv_group,
                                             privileged_groups=priv_group)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
      metric_orig_test.mean_difference())

# change labels to predictions
X_train_new.labels = pred_train
X_test_new.labels = pred_test

metric_orig_train = BinaryLabelDatasetMetric(X_train_new,
                                             unprivileged_groups=unpriv_group,
                                             privileged_groups=priv_group)
print(("#### After training"))
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
      metric_orig_train.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(X_test_new,
                                             unprivileged_groups=unpriv_group,
                                             privileged_groups=priv_group)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
      metric_orig_test.mean_difference())


# baseline = load_model("./models/baseline.h5")

# print()
# print('old model')

# X_test = X_test.drop(['intersection'], axis=1)
# X_train =X_train.drop(['intersection'], axis=1)
# X_val= X_val.drop(['intersection'], axis=1)

# pred_train = baseline.predict(X_train)
# pred_val = baseline.predict(X_val)
# pred_test = baseline.predict(X_test)

# metrics_train = Baseline.get_metrics(y_train, pred_train)
# print('accuracy:', metrics_train['accuracy'], 'f1_score: ', metrics_train['f1'],   ' matrix: ', metrics_train['matrix'])
# metrics_val = Baseline.get_metrics(y_val, pred_val)
# print('accuracy:', metrics_val['accuracy'], 'f1_score: ', metrics_val['f1'],  ' matrix: ', metrics_val['matrix'])
# metrics_test = Baseline.get_metrics(y_test, pred_test)
# print('accuracy:', metrics_test['accuracy'],'f1_score: ', metrics_test['f1'],  ' matrix: ', metrics_test['matrix'])
