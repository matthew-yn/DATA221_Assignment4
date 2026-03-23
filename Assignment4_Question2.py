from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

#Creates decision tree with entropy.
dt = DecisionTreeClassifier(criterion="entropy")
#Trains model.
dt.fit(X_train, y_train)

#Predictions on training and testing data.
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

#Computes accuracy.
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

#Prints accuracy.
print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

'''
Entropy measures uncertainty in the data.

The results of the training and testing accuracy suggests that the model is overfitting as the 
training accuracy is higher than the test accuracy.
'''