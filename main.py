import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
x = np.load("input_data/x.npy")
y = np.load("input_data/y.npy")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
x_train, x_test = x_train/255.0, x_test/255.0

input_shape = (28, 28, 1)

model = tf.keras.Sequential([
    Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
model.fit(x_train, y_train, epochs=5, batch_size=128)

model.save('model.h5')

loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print("Loss: ", loss, "Acc: ", acc)
predictions = model.predict(x_test)
print(y_test[0], predictions[0])
