from tensorflow import keras
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import *
from keras.layers import *


X, Y = [], []

WIDTH = 128
HEIGHT = 48

print("Loading image paths...")
all_image_paths = []
for i in range(10):
    root = './leapGestRecog/0' + str(i)
    dirs = os.listdir(root)
    for dir in dirs:
        path = root + '/' + dir
        img_paths = os.listdir(path)
        for p in img_paths:
            all_image_paths.append(path + '/' + p)

print("Loading images...")

for path in all_image_paths:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (WIDTH, HEIGHT))
    X.append(img)

    label = int(path.split('/')[3].split('_')[0][1])
    Y.append(label)

X = np.array(X, dtype='uint8')
X = X.reshape(len(all_image_paths), HEIGHT, WIDTH, 1)
Y = np.array(Y)

print('Images loaded: ', len(X))
print('Labels loaded: ', len(Y))


ts = 0.3
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=ts, random_state=42)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(HEIGHT, WIDTH, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy: %.2f" % (test_acc * 100))



