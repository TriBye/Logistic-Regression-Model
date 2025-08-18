import csv, math, json, os, random

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

learn_rate = 0.1
epochs = 1000
n = len(y)
eps = 1e-15

for epoch in range(epochs):
    error = []
    loss = 0.0

    for xi, yi in zip(X, y):
        z = sum(xi[j] * w[j] for j in range(len(w))) + b
        y_hat = 1.0 / (1.0 + math.exp(-z))
        error_i = y_hat - yi
        error.append(error_i)
        yp = min(max(y_hat, eps), 1 - eps)
        loss += -(yi * math.log(yp) + (1 - yi) * math.log(1 - yp))

    loss /= n

    if epoch == 0:
        old_loss = loss

    grad_w = [0.0] * len(w)
    grad_b = 0.0

    for i, xi in enumerate(X):
        e = error[i]
        for j in range(len(w)):
            grad_w[j] -= e * xi[j] / n
        grad_b -= e / n

    for j in range(len(w)):
        w[j] += learn_rate * grad_w[j]
    b += learn_rate * grad_b

    if epoch % 100 == 0 or epoch == epochs - 1:
        print("--------------------")
        print(f"Epoch: #{epoch}")
        print(f"LogLoss: {loss}")

print("\n--Training--")
print(f"Old Loss: {old_loss}")
print(f"New Loss: {loss}")
print(f"diff: {old_loss - loss}")

with open(weights_file, "w") as f:
    json.dump({"weights": w, "bias": b}, f)