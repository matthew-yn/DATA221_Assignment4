import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix

#Loads dataset.
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#Normalizes the pixel values to range [0,1].
X_train = X_train / 255.0
X_test = X_test / 255.0

#Reshape images to include the channel dimensions.
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

#Builds CNN model with Conv2D layer, MaxPooling2D layer, and Dense output layer.
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

#Compile model.
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#Train model.
model.fit(X_train, y_train_cat, epochs=15, batch_size=64, verbose=0)

#Predictions on testing data.
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

#Creates confusion matrix.
cm = confusion_matrix(y_test, y_pred)
#Prints confusion matrix.
print("Confusion Matrix:")
print(cm)

#Finds misclassified items.
misclassified = np.where(y_pred != y_test)[0]

#List of class names.
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Displays 3 misclassified images.
plt.figure(figsize=(10, 4))
for i in range(3):
    idx = misclassified[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28,28), cmap='grey')
    plt.title(f"True: {class_names[y_test[idx]]}\nPredicted: {class_names[y_pred[idx]]}")
    plt.axis('off')
plt.show()

'''
One pattern observed in the misclassifications are that they occur between similar clothing items.

One way to improve performance would be to add more convolution layers to help the model learn more
detailed features and differentiate between similar clothing items.
'''