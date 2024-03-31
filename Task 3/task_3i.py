import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = datagen.flow_from_directory(
        "D:\\Internship Projrcts\\Prodigy Infotech\\Task 3\\train\\train",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary"
      )

datagen1 = ImageDataGenerator(rescale=1./255)

test_set = datagen1.flow_from_directory(
        "D:\\Internship Projrcts\\Prodigy Infotech\\Task 3\\test1\\test1",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary"
      )

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,padding="same",kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation              ='linear'))

cnn.summary()

cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])

r=cnn.fit(x = training_set, validation_data = test_set, epochs = 15)

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

from tensorflow.keras.models import load_model

cnn.save('./classification.h5')

from tensorflow.keras.preprocessing import image
test_image = image.load_img("D:\\Internship Projrcts\\Prodigy Infotech\\Task 3\\train\\train\\cat.1028.jpg", target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

if result[0]<0:
    print("The image classified is cat")
else:
    print("The image classified is dog")

from tensorflow.keras.preprocessing import image
test_image = image.load_img("D:\\Internship Projrcts\\Prodigy Infotech\\Task 3\\train\\train\\dog.1077.jpg", target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)

if result[0]<0:
    print("The image classified is cat")
else:
    print("The image classified is dog")
