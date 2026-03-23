from sklearn.datasets import load_breast_cancer
import pandas as pd
#Loads breast cancer dataset from sklearn.
data = load_breast_cancer()

#Store feature matrix.
X = data.data
#Stores target labels.
y = data.target

#Prints shape of feature matrix and target labels.
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
#Prints number of samples that belong to each class.
print("Class distribution:")
print(pd.Series(y).value_counts())

'''
The dataset is slightly imbalanced because there are more benign cases than malignant cases.

Class balance is important because the model may become biased toward predicting that class 
more frequently if it has more samples. This may cuase the accuracy misleading, particularly
if the minority class is given more important.
'''