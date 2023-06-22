import matplotlib.pyplot as plt
import numpy as np
import sklearn
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from tensorflow.keras.models import load_model

from data_loading import data_preparation
from evaluation import UU_PAL
from NN import Baseline

# Load baseline model from disk
baseline = load_model("./models/baseline.h5")

# Load data from disk
fps = ["./data/intersectional_train.csv", "./data/intersectional_val.csv", "./data/intersectional_test.csv"]
X_train, X_val, X_test, y_train, y_val, y_test = data_preparation(fps=fps)
labels_intersection = {11: "Married Men", 12: "Single Men", 21: "Married Women", 22: "Single Women"}

# Equalised odds
# Equal TPR and FPR for each group
# TPR = TP / (TP + FN) = p[D = 1 | Y = 1 | A = a] for all four intersectional groups a
# FPR = FP / (FP + TN) = p[D = 1 | Y = 0 | A = a] for all four intersectional groups a

# Calculate and plot ROC curves for each group
y_trues = []
metrics_baseline = {"f1": [], "SP": [], "EO": [], "FPR": []}
fig, ax = plt.subplots(figsize=(6, 6))
for c, group in enumerate((11, 12, 21, 22)):
    idx = X_test["intersection"] == group

    X_subset = X_test[idx]
    y_true = y_test[idx]
    y_trues.append(y_true)

    # Drop intersectional group column
    X_subset = X_subset.drop("intersection", axis=1)

    # Predict on subset
    y_pred = baseline.predict(X_subset)

    # Calculate metrics for subset
    metrics = Baseline.get_metrics(y_true, y_pred, threshold=0.5)
    metrics_baseline["f1"].append(metrics["f1"])  # Fidelity metric
    metrics_baseline["SP"].append(metrics["SP"])  # Statistical parity
    metrics_baseline["EO"].append(metrics["EO"])  # Equalised opportunity, same as TPR
    metrics_baseline["FPR"].append(metrics["FPR"])  # False positive rate

    # Calculate ROC curve
    FPR, TPR, thresholds = sklearn.metrics.roc_curve(y_true, y_pred, pos_label=0)  # pos_label=0 because favourable is 0, e.g. no default
    ax.plot(FPR, TPR, label=labels_intersection[group], color=UU_PAL[c])  # Plot ROC curve

print("Baseline model metrics for each group:")
print(list(labels_intersection.values()))
for key in metrics_baseline.keys():
    metrics_baseline[key] = np.array(metrics_baseline[key])
    print(f"{key}: {metrics_baseline[key].round(2)}")

# Add overall ROC curve
y_pred = baseline.predict(X_test.drop("intersection", axis=1))  # Predict on all test data
FPR, TPR, thresholds = sklearn.metrics.roc_curve(y_test, y_pred, pos_label=0)  # pos_label=0 because favourable is 0, e.g. no default
ax.plot(FPR, TPR, label="Overall", color="black", linestyle="--")

ax.set_aspect("equal", "box")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC curve on test data for baseline model")
plt.legend(title="Intersectional group")
plt.savefig("./images/ROC_baseline.eps")
plt.show()

# Apply equalised odds via AIF360
# Create post-processing object
eq_odds = EqOddsPostprocessing(unprivileged_groups=[{"intersection": 11}, {"intersection": 12}],
                               privileged_groups=[{"intersection": 21}, {"intersection": 22}], seed=42)

# Combine X and y
aif_train = X_train.copy()
aif_train["target_column"] = y_train

aif_test = X_test.copy()
aif_test["target_column"] = y_test

# Convert to BinaryLabelDataset
# Favourable is to not default (0), unfavourable is to default (1)
aif_train = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=aif_train,
                               label_names=["target_column"], protected_attribute_names=["intersection"])
aif_test = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=aif_test,
                              label_names=["target_column"], protected_attribute_names=["intersection"])

# Predict on training data as preparation for training eq odds model
X_train = X_train.drop("intersection", axis=1)
y_train_pred = baseline.predict(X_train)
aif_train_pred = aif_train.copy(deepcopy=True)  # Dataset with predicted labels
aif_train_pred.labels = y_train_pred

X_test = X_test.drop("intersection", axis=1)
y_test_pred = baseline.predict(X_test)
aif_test_pred = aif_test.copy(deepcopy=True)  # Dataset with predicted labels
aif_test_pred.labels = y_test_pred

# Fit eq odds on train
eq_odds.fit(aif_train, aif_train_pred)  # Fit on original vs predicted labels by baseline model

# Predict on test
eq_odds_test = eq_odds.predict(aif_test_pred)

# Calculate new TPR and FPR for each group
metrics_new = {"f1": [], "SP": [], "EO": [], "FPR": []}
fig, ax = plt.subplots(figsize=(6, 6))
for c, group in enumerate((11, 12, 21, 22)):
    idx = eq_odds_test.protected_attributes[:, 0] == group
    y_pred = eq_odds_test.labels[idx]

    # Calculate metrics for subset
    metrics = Baseline.get_metrics(y_trues[c], y_pred, threshold=0.5)
    metrics_new["f1"].append(metrics["f1"])  # Fidelity metric
    metrics_new["SP"].append(metrics["SP"])  # Statistical parity
    metrics_new["EO"].append(metrics["EO"])  # Equalised opportunity, same as TPR
    metrics_new["FPR"].append(metrics["FPR"])  # False positive rate

    # Calculate ROC curve
    # pos_label=0 because favourable is 0, e.g. no default
    FPR, TPR, thresholds = sklearn.metrics.roc_curve(y_trues[c], y_pred, pos_label=0)
    ax.plot(FPR, TPR, label=labels_intersection[group], color=UU_PAL[c])  # Plot ROC curve

print("After Equalised Odds mitigation metrics for each group:")
print(list(labels_intersection.values()))
for key in metrics_new.keys():
    metrics_new[key] = np.array(metrics_new[key])
    print(f"{key}: {metrics_new[key].round(2)}")

weights = [y_trues[c].size / y_test.size for c in range(4)]  # Weights for micro-average
# Calculate percentual change of metrics, element-wise
for metric, old_value in metrics_baseline.items():
    change = (metrics_new[metric] - old_value) / old_value  # Percental change
    macro = np.mean(change)
    micro = np.average(change, weights=weights)
    print(f"Change in {metric} scores: {change.round(2)}; Macro-average: {macro.round(2)}; Micro-average: {micro.round(2)}")

# Add overall ROC curve
# pos_label=0 because favourable is 0, e.g. no default
FPR, TPR, thresholds = sklearn.metrics.roc_curve(y_test, eq_odds_test.labels, pos_label=0)
ax.plot(FPR, TPR, label="Overall", color="black", linestyle="--")

ax.set_aspect("equal", "box")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC curve on test data for baseline model with equalised odds")
plt.legend(title="Intersectional group")
plt.savefig("./images/ROC_eq_odds.eps")
plt.show()
