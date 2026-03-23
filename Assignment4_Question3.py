import pandas as pd
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

#Creates constrained decision tree with entropy.
dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
#Train model.
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

#Gets feature importance values.
importance = dt.feature_importances_
feature_names = data.feature_names

#Creates DataFrame to display feature importance.
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

#Prints top five most important features.
print("Top 5 Features:")
print(importance_df.head(5))

'''
Controlling model complexity reduces overfitting and improves generalization of the model.

Feature importance helps interpret which variables influence predictions the most.
'''