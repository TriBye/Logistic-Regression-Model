import csv
import random
import math

random.seed(42)

num_samples = 200
w1 = 2.0
w2 = -1.5
bias = -1.0

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["y", "x1", "x2"])
    for _ in range(num_samples):
        x1 = random.uniform(-4, 4)
        x2 = random.uniform(-4, 4)
        linear_comb = w1 * x1 + w2 * x2 + bias
        prob = sigmoid(linear_comb)
        y = 1 if random.random() < prob else 0
        writer.writerow([y, x1, x2])