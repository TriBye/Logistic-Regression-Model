import csv, math, random, json, os

#load dataset
with open("dataset.csv", "r") as f:
    reader = csv.reader(f)
    dataset = [[float(x) for x in row] for row in reader]

#define X and y
X = [row[1:] for row in dataset]
y = [row[0] for row in dataset]

#load weights and bias
weights_file = "weights.json"
if os.path.exists(weights_file):
    with open(weights_file, "r") as f:
        params = json.load(f)
    w = params["weights"]
    b = params["bias"]
else:
    w = [random.uniform(-0.01, 0.01) for _ in range(len(X[0]))]
    b = random.uniform(-0.01, 0.01)

#prediction
preds = []
for row in X:
    z = sum(row[i] * w[i] for i in range(len(row))) + b
    y_p = 1 / (1 + math.exp(-z))
    preds.append(y_p)

#accuracy and feedback
threshold = 0.5
acc = 0
for i in range(len(preds)):
    pred_class = 1 if preds[i] > threshold else 0
    print(f"Probability: {preds[i]:.6f} Classification: {pred_class} Class: {int(y[i])}")
    if pred_class == int(y[i]):
        acc += 1

print(f"Accuracy: {acc / len(y):.4f}")