import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.normalization import BatchNormalization

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.Sequential()
initializer = 'he_normal'

model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(512, activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax',
          kernel_initializer=initializer))
adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=50, epochs=30)
model.evaluate(x_test, y_test)
