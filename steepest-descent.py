import numpy as np

# Dataset
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
y = np.array([0.4, 0.8, 1.3, 0.7, 1.5, 1.1, 0.5, 0.2, 0.7, 0.8, 1.3, 1.0], dtype=float)

# Model:
# y_hat = w * x
# No bias is used.

def predict(w, x):
    return w * x

def loss(w, x, y):
    y_hat = predict(w, x)
    return np.sum((y_hat - y) ** 2)

def gradient(w, x, y):
    y_hat = predict(w, x)
    return 2 * np.sum((y_hat - y) * x)

def dynamic_alpha(w, x, y):
    """
    Derivation:
    alpha = sum(x_i * (w*x_i - y_i)) / (g * sum(x_i^2))

    Because:
    g = 2 * sum(x_i * (w*x_i - y_i))

    For one attribute and no bias:
    alpha = 1 / (2 * sum(x_i^2))
    """

    g = gradient(w, x, y)

    if abs(g) < 1e-12:
        return 0.0

    numerator = np.sum(x * (w * x - y))
    denominator = g * np.sum(x ** 2)

    return numerator / denominator

def steepest_descent(x, y, w0=0.0, max_iter=100, tolerance=1e-12):
    w = w0
    history = []

    for iteration in range(max_iter):
        current_loss = loss(w, x, y)
        grad = gradient(w, x, y)

        if abs(grad) < tolerance:
            history.append({
                "iteration": iteration,
                "w": w,
                "loss": current_loss,
                "gradient": grad,
                "alpha": 0.0
            })
            break

        alpha = dynamic_alpha(w, x, y)

        history.append({
            "iteration": iteration,
            "w": w,
            "loss": current_loss,
            "gradient": grad,
            "alpha": alpha
        })

        w_new = w - alpha * grad

        if abs(w_new - w) < tolerance:
            w = w_new
            break

        w = w_new

    return w, history


w_sd, history_sd = steepest_descent(
    x,
    y,
    w0=0.0
)

print("Steepest Descent Result")
print("Final w:", w_sd)
print("Final loss:", loss(w_sd, x, y))
print("Final equation: y_hat =", w_sd, "* x")

print("\nIterations:")
for row in history_sd:
    print(row)