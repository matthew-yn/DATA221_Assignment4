from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

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

#Computes accuracy.
test_acc = model.evaluate(X_test, y_test_cat)
#Prints accuracy.
print("Test accuracy:", test_acc)

'''
CNNs are preferred for image data because they can learn spatial patterns such as
edges, textures, and shapes using shared filters.

The convolution layer detects edges, textures, and shapes that help it to classify
and identify clothing items.
'''