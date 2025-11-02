from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalization
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2])).astype("float32")/255
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2])).astype("float32")/255

#one-hot encoding 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Creating ANN model
model = Sequential()
model.add(Dense(512, activation="relu", input_shape= (28*28,)))
model.add(Dense(256, activation="tanh"))
#output layer
model.add(Dense(10, activation="softmax"))

model.summary()

#Compiling the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#Defining callbacks and ANN training
earlyStopping = EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True)
checkpoint = ModelCheckpoint("ann_best_model.h5",monitor="val_loss",save_best_only=True)

history = model.fit(x_train,y_train,
          epochs=10,
          batch_size=64,
          validation_split=0.2,
          callbacks=[earlyStopping,checkpoint])

#Model evaluation 
test_loss, test_acc = model.evaluate(x_test,y_test)

#visualization
plt.figure()
plt.plot(history.history["accuracy"], marker = "o", label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], marker = "o", label = "Validation Accuracy")
plt.title("ANN Accuracy on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(history.history["loss"], marker = "o", label = "Training Loss")
plt.plot(history.history["val_loss"], marker = "o", label = "Validation Loss")
plt.title("ANN Loss on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
