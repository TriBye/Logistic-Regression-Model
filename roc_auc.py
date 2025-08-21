#!/usr/bin/env python3
import csv, json, math, os, random
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def load_data(dataset_path):
    with open(dataset_path, "r") as f:
        reader = csv.reader(f)
        data = [[float(x) for x in row] for row in reader]
    X = [row[1:] for row in data]
    y = [int(row[0]) for row in data]
    return X, y

def load_params(weights_path, n_features):
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            params = json.load(f)
        w = params["weights"]
        b = params["bias"]
        return w, b
    else:
        w = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
        b = random.uniform(-0.01, 0.01)
        return w, b

def predict_proba(X, w, b):
    preds = []
    for row in X:
        z = sum(row[i] * w[i] for i in range(len(w))) + b
        preds.append(sigmoid(z))
    return preds

def roc_auc(y_true, y_score):
    pairs = sorted(zip(y_true, y_score), key=lambda t: t[1], reverse=True)
    P = sum(1 for y in y_true if y == 1)
    N = sum(1 for y in y_true if y == 0)
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    for yi, _ in pairs:
        if yi == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P if P else 0.0)
        fpr.append(fp / N if N else 0.0)
    auc = 0.0
    for i in range(len(fpr) - 1):
        auc += (fpr[i+1] - fpr[i]) * (tpr[i] + tpr[i+1]) / 2.0
    return fpr, tpr, auc

def main():
    dataset_path = "dataset.csv"
    weights_path = "weights.json"
    X, y = load_data(dataset_path)
    w, b = load_params(weights_path, len(X[0]))
    y_score = predict_proba(X, w, b)
    fpr, tpr, auc = roc_auc(y, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=160)

    print(f"AUC: {auc:.6f}")
    print("ROC curve saved to roc_curve.png")

if __name__ == "__main__":
    main()