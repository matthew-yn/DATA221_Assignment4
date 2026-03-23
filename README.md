DATA221 - Assignment 4

Assignment4_Question1.py
Loads the Breast Caner Wisconsin dataset and constructs feature matrix X and target vector y. The program then finds and prints the shapes and the class distribution of the dataset.

Assignment4_Question2.py
Performs an 80/20 stratified train-test split then trains a decision tree classifier using entropy. The program then computes and prints the training and test accuracy.

Assignment4_Question3.py
Trains a constrained Decision Tree using the same 80/20 stratified train-test split. The program then computes the training and test accuracy and finds the top five most important features and prints them out.

Assignment4_Question4.py
Standardizes the input features then builds and trains a neural network with a hidden layer and a Sigmoid output unit using the 80/20 stratified train-test split. The program then computes and prints the training and test accuracy.

Assignment4_Question5.py
Creates confusion matrices for the constrained decision tree and neural network from questions 3 and 4 then compares the two models.

Assignment4_Question6.py
Loads Fashion MNIST and normalizes pixel values to range [0,1] and reshapes images to include the channel dimesion. The program then builds a CNN with a Conv2D layer, a MaxPooling2D layer, and a Dense output layer. It trains the model for at least fifteen epochs and computes and prints the test accuracy.

Assignment4_Question7.py
Uses the CNN model from question 6 to make predictions on the test set then creates and prints the confusion matrix. The program then identifies and visualizes three misclassified images, showing the true and predicted labels.
