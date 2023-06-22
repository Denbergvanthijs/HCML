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

# Equalised odds
# Equal TPR and FPR for each group
# TPR = TP / (TP + FN) = p[D = 1 | Y = 1 | A = a] for all four intersectional groups a
# FPR = FP / (FP + TN) = p[D = 1 | Y = 0 | A = a] for all four intersectional groups a

# Calculate current TPR and FPR for each group
F1 = []
SP = []  # Statistical parity
EO = []  # Equal opportunity
FPR = []  # False positive rate
y_subsets = []
y_scores = []
labels_intersection = {11: "Married Men", 12: "Single Men", 21: "Married Women", 22: "Single Women"}
for group in (11, 12, 21, 22):
    idx = X_test["intersection"] == group

    X_subset = X_test[idx]
    y_subset = y_test[idx]
    y_subsets.append(y_subset)

    # Drop intersectional group column
    X_subset = X_subset.drop("intersection", axis=1)

    # Predict on subset
    y_pred = baseline.predict(X_subset)
    y_scores.append(y_pred)

    # Calculate metrics for subset
    metrics = Baseline.get_metrics(y_subset, y_pred, threshold=0.5)

    # Fidelity metrics
    F1.append(metrics["f1"])

    # Fairness metrics
    SP.append(metrics["SP"])
    EO.append(metrics["EO"])  # EO is equal to TPR/Recall
    FPR.append(metrics["FPR"])

# To numpy array
F1 = np.array(F1)
SP = np.array(SP)
EO = np.array(EO)
FPR = np.array(FPR)

print("Baseline model metrics for each group:")
print(list(labels_intersection.values()))
print(f"F1: {F1.round(2)}")
print(f"SP: {SP.round(2)}")
print(f"EO/TPR: {EO.round(2)}")
print(f"FPR: {FPR.round(2)}")

fig, ax = plt.subplots(figsize=(6, 6))
# Calculate and plot ROC curves for each group
tpr_scores = []
fpr_scores = []
for c, group in enumerate((11, 12, 21, 22)):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_subsets[c], y_scores[c])
    tpr_scores.append(tpr)
    fpr_scores.append(fpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, label=labels_intersection[group], color=UU_PAL[c])

# Add overall ROC curve
X_test_no_inter = X_test.drop("intersection", axis=1)
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, baseline.predict(X_test_no_inter))
ax.plot(fpr, tpr, label="Overall", color="black", linestyle="--")

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
F1_new = []
SP_new = []  # Statistical parity
EO_new = []  # Equal opportunity
FPR_new = []
for c, group in enumerate((11, 12, 21, 22)):
    idx = eq_odds_test.protected_attributes[:, 0] == group
    y_pred = eq_odds_test.labels[idx]

    # Calculate metrics for subset
    metrics = Baseline.get_metrics(y_subsets[c], y_pred, threshold=0.5)

    # Fidelity metrics
    F1_new.append(metrics["f1"])

    # Fairness metrics
    SP_new.append(metrics["SP"])
    EO_new.append(metrics["EO"])  # EO is equal to TPR/Recall
    FPR_new.append(metrics["FPR"])

# To numpy array
F1_new = np.array(F1_new)
SP_new = np.array(SP_new)
EO_new = np.array(EO_new)
FPR_new = np.array(FPR_new)

print("After Equal Odds mitigation metrics for each group:")
print(list(labels_intersection.values()))
print(f"F1: {F1.round(2)}")
print(f"SP: {SP.round(2)}")
print(f"EO/TPR: {EO.round(2)}")
print(f"FPR: {FPR.round(2)}")

# Calculate percentual change of metrics, element-wise
weights = [y_subsets[c].size / y_test.size for c in range(4)]
for new, old, name in zip((F1_new, SP_new, EO_new, FPR_new), (F1, SP, EO, FPR), ("F1", "SP", "EO", "FPR")):
    change = (new - old) / old
    macro = np.mean(change)
    micro = np.average(change, weights=weights)
    print(f"Change in {name} scores: {change.round(2)}; Macro-average: {macro.round(2)}; Micro-average: {micro.round(2)}")

# New ROC curve
fig, ax = plt.subplots(figsize=(6, 6))
# Calculate and plot ROC curves for each group
tpr_scores_new = []
fpr_scores_new = []
for c, group in enumerate((11, 12, 21, 22)):
    idx = eq_odds_test.protected_attributes[:, 0] == group
    y_pred = eq_odds_test.labels[idx]

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_subsets[c], y_pred)
    tpr_scores_new.append(tpr)
    fpr_scores_new.append(fpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, label=labels_intersection[group], color=UU_PAL[c])

# Add overall ROC curve
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, eq_odds_test.labels)
ax.plot(fpr, tpr, label="Overall", color="black", linestyle="--")

ax.set_aspect("equal", "box")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC curve on test data for baseline model with equalised odds")
plt.legend(title="Intersectional group")
plt.savefig("./images/ROC_eq_odds.eps")
plt.show()