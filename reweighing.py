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

def categorise(row, weights, intersection):  
    """function for creating a new column with weights"""
    if(intersection):
        if row['intersection'] == 11 and row['y'] == 1:
            return weights[7]
        elif row['intersection'] == 12 and row['y'] == 1:
            return weights[6]
        elif row['intersection'] == 21  and row['y'] == 1:
            return weights[5]
        elif row['intersection'] == 22  and row['y'] == 1:
            return weights[4]
        elif row['intersection'] == 11  and row['y'] == 0:
            return weights[3]
        elif row['intersection'] == 12  and row['y'] == 0:
            return weights[2]
        elif row['intersection'] == 21  and row['y'] == 0:
            return weights[1]
        else:
            return weights[0]
    else:
        if row['intersection'] == 11 and row['y'] == 1:
            return weights[3]
        elif row['intersection'] == 12 and row['y'] == 1:
            return weights[3]
        elif row['intersection'] == 21  and row['y'] == 1:
            return weights[2]
        elif row['intersection'] == 22  and row['y'] == 1:
            return weights[2]
        elif row['intersection'] == 11  and row['y'] == 0:
            return weights[1]
        elif row['intersection'] == 12  and row['y'] == 0:
            return weights[1]
        elif row['intersection'] == 21  and row['y'] == 0:
            return weights[0]
        else:
            return weights[0]
    
#reweighing function for gender and defaulting
def Reweighing_data(protected_attribute, classification_class, dataset, intersection=True):

    total_data= dataset.join(classification_class)
    default_list = classification_class.value_counts(normalize=True)
    P_intersection = protected_attribute.value_counts(normalize=True)
    P_intersection=P_intersection.values

    if intersection == False: P_intersection = np.asarray([P_intersection[i*2] + P_intersection[i*2+1] for i in range(2)])

    # Calculate P expected
    P_default = default_list.values[1]
    P_no_default = default_list.values[0]
    P_exp_default = P_intersection*P_default
    P_exp_nodefault =P_intersection*P_no_default
    P_exp_values = np.concatenate((P_exp_nodefault, P_exp_default))

    # Calculate P observed
    counts_observed = total_data.value_counts(['intersection', 'y'], normalize=True)
    P_observed = counts_observed.values
    if intersection == False: P_observed = np.asarray([P_observed[i*2] +P_observed[i*2+1] for i in range(4)])

    weights = P_exp_values/P_observed #from 22-0, 21-0 to 12-1, 11-1 (so we start with single females no default)
    weights = total_data.apply(lambda row: categorise(row, weights, intersection), axis=1)
    return weights  

def apply_reweighing(own_function=True, intersection=True):
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
    
    weights=None
    if(own_function):
        weights = Reweighing_data(X_train['intersection'], y_train, X_train, intersection=intersection)
    else:
        data_reweighed = Reweighing(unprivileged_groups=unpriv_group,privileged_groups=priv_group).fit_transform(X_train_new)
        weights = data_reweighed.instance_weights

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
    print(("#### Before training"))
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
apply_reweighing()

