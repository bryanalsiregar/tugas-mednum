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

def gradient_descent(x, y, w0=0.0, learning_rate=0.001, max_iter=10000, tolerance=1e-12):
    w = w0
    history = []

    for iteration in range(max_iter):
        current_loss = loss(w, x, y)
        grad = gradient(w, x, y)

        history.append({
            "iteration": iteration,
            "w": w,
            "loss": current_loss,
            "gradient": grad
        })

        if abs(grad) < tolerance:
            break

        w_new = w - learning_rate * grad

        if abs(w_new - w) < tolerance:
            w = w_new
            break

        w = w_new

    return w, history


w_gd, history_gd = gradient_descent(
    x,
    y,
    w0=0.0,
    learning_rate=0.001
)

print("Gradient Descent Result")
print("Final w:", w_gd)
print("Final loss:", loss(w_gd, x, y))
print("Final equation: y_hat =", w_gd, "* x")

print("\nFirst 10 iterations:")
for row in history_gd[:10]:
    print(row)