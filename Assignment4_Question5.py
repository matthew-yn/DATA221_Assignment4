from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
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

#Constrained Decision Tree

#Creates constrained decision tree with entropy.
dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
#Train model.
dt.fit(X_train, y_train)

#Predictions on testing data.
y_pred_dt = dt.predict(X_test)

#Creates decision tree confusion matrix.
cm_dt = confusion_matrix(y_test, y_pred_dt)

#Prints decision tree confusion matrix.
print("Decision Tree Confusion Matrix:")
print(cm_dt)

#Neural Network.

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
model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

#Predictions on testing data.
y_pred_nn = (model.predict(X_test_scaled) >= 0.5).astype(int)

#Creates neural network confusion matrix.
cm_nn = confusion_matrix(y_test, y_pred_nn)

#Prints neural network confusion matrix.
print("Neural Network Confusion Matrix:")
print(cm_nn)

'''
The neural network would be preferred for this task because it better generalizes 
the data as it can learn more complex patterns from the data.

The decision tree is easy to interpret but may underfit/overfit.
The neural network can capture complex patterns but is more difficult to interpret.
'''