import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 32
epochs = 10

x = np.load("input_data/x.npy")
y = np.load("input_data/y.npy")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           zoom_range=0.25,
                                           brightness_range=[0.5, 1.5]
                                           )
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
val_data_gen = validation_image_generator.flow(x_test, y_test, batch_size=batch_size)


# x_train, x_test = x_train/255.0, x_test/255.0

input_shape = (28, 28, 1)

model = tf.keras.Sequential([
    Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_shape, use_bias=True),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(32, kernel_size=(3, 3), activation='relu', use_bias=True),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', use_bias=True),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu', use_bias=True),
    Dropout(0.2),
    Dense(128, activation='relu', use_bias=True),
    Dense(10, activation=tf.nn.softmax, use_bias=True)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=len(x_test) // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('model.h5')

loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Loss: ", loss, "Acc: ", acc)
predictions = model.predict(x_test)
print(y_test[0], predictions[0])
