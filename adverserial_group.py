import tensorflow.compat.v1 as tf
from data_loading import data_preparation
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from NN import Baseline
import sys

# error debug
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

tf.disable_eager_execution()


# PARAMS
total_epoch = 10

##

# https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/inprocessing/adversarial_debiasing.py
# Load data from disk
fps = ["./data/intersectional_train.csv",
       "./data/intersectional_val.csv", "./data/intersectional_test.csv"]
X_train, X_val, X_test, y_train, y_val, y_test = data_preparation(fps=fps)

# Calculate current TPR and FPR for each group
labels_intersection = {11: "Married Men", 12: "Single Men",
                       21: "Married Women", 22: "Single Women"}

all_groups = set([11, 12, 21, 22])
metrics_base = {"f1": [], "SP": [], "EO": [], "FPR": []}
metrics_adv = {"f1": [], "SP": [], "EO": [], "FPR": []}

for curr_group, _ in labels_intersection.items():

    other_group = all_groups.difference(set([curr_group]))

    X_train, X_val, X_test, y_train, y_val, y_test = data_preparation(fps=fps)

    X_train["intersection"].replace(other_group, 0, inplace=True)
    X_train["intersection"].replace(curr_group, 1, inplace=True)

    X_test["intersection"].replace(other_group, 0, inplace=True)
    X_test["intersection"].replace(curr_group, 1, inplace=True)

    unprivileged_groups = [{"intersection": 0}]
    privileged_groups = [{"intersection": 1}]

    # Combine X and y
    aif_train = X_train.copy()
    aif_train["target_column"] = y_train

    aif_test = X_test.copy()
    aif_test["target_column"] = y_test

    # Convert to BinaryLabelDataset
    # Favourable is to not default (0), unfavourable is to default (1)
    dataset_orig_train = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=aif_train,
                                            label_names=["target_column"], protected_attribute_names=["intersection"])
    dataset_orig_test = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=aif_test,
                                           label_names=["target_column"], protected_attribute_names=["intersection"])

    # Metric for the original dataset
    orig_stdout = sys.stdout
    f = open(f'results/{labels_intersection[curr_group]}/org_data.txt', 'w+')
    sys.stdout = f

    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    print(("#### Original training dataset"))
    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_orig_train.mean_difference())
    metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_orig_test.mean_difference())

    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(
        dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(
        dataset_orig_test.features)
    metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                   unprivileged_groups=unprivileged_groups,
                                                   privileged_groups=privileged_groups)

    print("\n#### Scaled dataset - Verify that the scaling does not affect the group label statistics")
    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_scaled_train.mean_difference())
    metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_scaled_test.mean_difference())

    # stop write to file
    sys.stdout = orig_stdout
    f.close()

    # Learn plan classifier without debiasing
    sess = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                       unprivileged_groups=unprivileged_groups,
                                       scope_name='plain_classifier',
                                       debias=False,
                                       sess=sess,
                                       num_epochs=total_epoch,
                                       batch_size=32,
                                       classifier_num_hidden_units=50)

    plain_model.fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_nodebiasing_train = plain_model.predict(dataset_orig_train)
    dataset_nodebiasing_test = plain_model.predict(dataset_orig_test)

    # write all to file
    orig_stdout = sys.stdout
    f = open(f'results/{labels_intersection[curr_group]}/org_model.txt', 'w+')
    sys.stdout = f

    # Metrics for the dataset from plain model (without debiasing)
    print(("#### Plain model - without debiasing - classification metrics"))
    classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                              dataset_nodebiasing_test,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" %
          classified_metric_nodebiasing_test.accuracy())
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
    print("Test set: Balanced classification accuracy = %f" %
          bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" %
          classified_metric_nodebiasing_test.disparate_impact())
    print("Test set: Equal opportunity difference = %f" %
          classified_metric_nodebiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" %
          classified_metric_nodebiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" %
          classified_metric_nodebiasing_test.theil_index())

    # Metrics for the dataset from plain model (without debiasing)
    print(("\n#### Plain model - without debiasing - dataset metrics"))
    metric_dataset_nodebiasing_train = BinaryLabelDatasetMetric(dataset_nodebiasing_train,
                                                                unprivileged_groups=unprivileged_groups,
                                                                privileged_groups=privileged_groups)

    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_dataset_nodebiasing_train.mean_difference())

    metric_dataset_nodebiasing_test = BinaryLabelDatasetMetric(dataset_nodebiasing_test,
                                                               unprivileged_groups=unprivileged_groups,
                                                               privileged_groups=privileged_groups)

    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_dataset_nodebiasing_test.mean_difference())

    print(("\n#### Plain model - without debiasing - classification metrics"))
    classified_metric_nodebiasing_test = ClassificationMetric(dataset_orig_test,
                                                              dataset_nodebiasing_test,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" %
          classified_metric_nodebiasing_test.accuracy())
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
    print("Test set: Balanced classification accuracy = %f" %
          bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" %
          classified_metric_nodebiasing_test.disparate_impact())
    print("Test set: Equal opportunity difference = %f" %
          classified_metric_nodebiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" %
          classified_metric_nodebiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" %
          classified_metric_nodebiasing_test.theil_index())
    # stop write to file
    sys.stdout = orig_stdout
    f.close()

    metrics = Baseline.get_metrics(
        dataset_orig_test.labels, dataset_nodebiasing_test.labels, threshold=None)
    metrics_base["f1"].append(metrics["f1"])  # Fidelity metric
    metrics_base["SP"].append(metrics["SP"])  # Statistical parity
    # Equalised opportunity, same as TPR
    metrics_base["EO"].append(metrics["EO"])
    metrics_base["FPR"].append(metrics["FPR"])  # False positive rate

    # Apply in-processing algorithm based on adversarial learning
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()

    # Learn parameters with debias set to True
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,
                                          unprivileged_groups=unprivileged_groups,
                                          scope_name='debiased_classifier',
                                          debias=True,
                                          sess=sess,
                                          num_epochs=total_epoch,
                                          batch_size=32,
                                          classifier_num_hidden_units=50)
    debiased_model.fit(dataset_orig_train)

    # Apply the plain model to test data
    dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

    orig_stdout = sys.stdout
    f = open(f'results/{labels_intersection[curr_group]}/debias.txt', 'w+')
    sys.stdout = f

    # Metrics for the dataset from plain model (without debiasing)
    print(("#### Plain model - without debiasing - dataset metrics"))
    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_dataset_nodebiasing_train.mean_difference())
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_dataset_nodebiasing_test.mean_difference())

    # Metrics for the dataset from model with debiasing
    print(("\n#### Model - with debiasing - dataset metrics"))
    metric_dataset_debiasing_train = BinaryLabelDatasetMetric(dataset_debiasing_train,
                                                              unprivileged_groups=unprivileged_groups,
                                                              privileged_groups=privileged_groups)

    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_dataset_debiasing_train.mean_difference())

    metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test,
                                                             unprivileged_groups=unprivileged_groups,
                                                             privileged_groups=privileged_groups)

    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" %
          metric_dataset_debiasing_test.mean_difference())

    print(("\n#### Plain model - without debiasing - classification metrics"))
    print("Test set: Classification accuracy = %f" %
          classified_metric_nodebiasing_test.accuracy())
    TPR = classified_metric_nodebiasing_test.true_positive_rate()
    TNR = classified_metric_nodebiasing_test.true_negative_rate()
    bal_acc_nodebiasing_test = 0.5*(TPR+TNR)
    print("Test set: Balanced classification accuracy = %f" %
          bal_acc_nodebiasing_test)
    print("Test set: Disparate impact = %f" %
          classified_metric_nodebiasing_test.disparate_impact())
    print("Test set: Equal opportunity difference = %f" %
          classified_metric_nodebiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" %
          classified_metric_nodebiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" %
          classified_metric_nodebiasing_test.theil_index())

    print(("\n#### Model - with debiasing - classification metrics"))
    classified_metric_debiasing_test = ClassificationMetric(dataset_orig_test,
                                                            dataset_debiasing_test,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" %
          classified_metric_debiasing_test.accuracy())
    TPR = classified_metric_debiasing_test.true_positive_rate()
    TNR = classified_metric_debiasing_test.true_negative_rate()
    bal_acc_debiasing_test = 0.5*(TPR+TNR)
    print("Test set: Balanced classification accuracy = %f" %
          bal_acc_debiasing_test)
    print("Test set: Disparate impact = %f" %
          classified_metric_debiasing_test.disparate_impact())
    print("Test set: Equal opportunity difference = %f" %
          classified_metric_debiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" %
          classified_metric_debiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" %
          classified_metric_debiasing_test.theil_index())
    # stop write to file
    sys.stdout = orig_stdout
    f.close()

    metrics = Baseline.get_metrics(
        dataset_orig_test.labels, dataset_debiasing_test.labels, threshold=None)
    metrics_adv["f1"].append(metrics["f1"])  # Fidelity metric
    metrics_adv["SP"].append(metrics["SP"])  # Statistical parity
    # Equalised opportunity, same as TPR
    metrics_adv["EO"].append(metrics["EO"])
    metrics_adv["FPR"].append(metrics["FPR"])  # False positive rate


orig_stdout = sys.stdout
f = open('results/group_metrics.txt', 'w+')
sys.stdout = f

print("Baseline model metrics for each group:")
print(list(labels_intersection.values()))
for key in metrics_base.keys():
    metrics_base[key] = np.array(metrics_base[key])
    print(f"{key}: {metrics_base[key].round(2)}")

print("\Adverserial model metrics for each group:")
print(list(labels_intersection.values()))
for key in metrics_adv.keys():
    metrics_adv[key] = np.array(metrics_adv[key])
    print(f"{key}: {metrics_adv[key].round(2)}")

    # stop write to file
sys.stdout = orig_stdout
f.close()
