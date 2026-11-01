import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

Y_train = np.eye(10)[y_train]
Y_test = np.eye(10)[y_test]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("Pixel range:", X.min(), "to", X.max())

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("One-hot training shape:", Y_train.shape)
print("One-hot test shape:", Y_test.shape)
print("First training label:", y_train[0])
print("First one-hot label:", Y_train[0])