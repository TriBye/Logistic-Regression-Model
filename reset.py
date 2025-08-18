import json, random

weights_file = "weights.json"

def reset_weights(n_features):
    w = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
    b = random.uniform(-0.01, 0.01)
    params = {"weights": w, "bias": b}
    with open(weights_file, "w") as f:
        json.dump(params, f)
    print("Weights und Bias zurückgesetzt.")

reset_weights(n_features=2)