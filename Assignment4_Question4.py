from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers

#Loads dataset.
data = load_breast_cancer()
#Stores feature matrix.
X = data.data
#Stores target labels.
y = data.target

#Splits into 80% training and 20% testing with stratification.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scales features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Builds neural network with hidden layer.
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
#Compiles model.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#Trains model.
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

#Computes accuracy.
train_acc = model.evaluate(X_train_scaled, y_train)
test_acc = model.evaluate(X_test_scaled, y_test)

#Prints accuracy.
print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

'''
Feature scaling is necessary for neural networks because it ensures all features contribute equally
to model learning and improves speed and stability.

An epoch is one full pass through the training dataset.
'''