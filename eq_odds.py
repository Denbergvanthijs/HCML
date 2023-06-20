import matplotlib.pyplot as plt
import numpy as np
import sklearn
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
TPR = []
FPR = []
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

    TPR.append(metrics["TP"] / (metrics["TP"] + metrics["FN"]))
    FPR.append(metrics["FP"] / (metrics["FP"] + metrics["TN"]))

# To numpy array
TPR = np.array(TPR)
FPR = np.array(FPR)

print("Old TPR and FPR for each group:")
print(list(labels_intersection.values()))
print(f"TPR: {TPR.round(2)}")
print(f"FPR: {FPR.round(2)}")

fig, ax = plt.subplots(figsize=(6, 6))
# Calculate and plot ROC curves for each group
for c, group in enumerate((11, 12, 21, 22)):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_subsets[c], y_scores[c])

    # Plot ROC curve
    ax.plot(fpr, tpr, label=labels_intersection[group], color=UU_PAL[c])

ax.set_aspect("equal", "box")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC Curve for Baseline Model")
plt.legend()
plt.savefig("./images/ROC_baseline.eps")
plt.show()
