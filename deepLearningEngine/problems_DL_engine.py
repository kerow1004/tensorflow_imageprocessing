import os

from PIL import Image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.utils import np_utils
from keras.optimizers import RMSprop


from keras.applications import VGG16
import tensorflow as tf
import numpy as np

probId = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
root_dir = "../trainingSet"
# for i in os.listdir(root_dir):
#     probId.append(i)
nb_classes = len(probId)

image_w = 64
image_h = 64

def main():
    X_train, X_test, y_train, y_test = np.load("../test_problem_makedata.npy")
    # 데이터 정규화하기
    X_train = X_train.astype("float32") / 256
    X_test  = X_test.astype("float32")  / 256

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = model_Train(X_train, Y_train, X_test, Y_test)
    model_evaluate(model, X_test, Y_test)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation(tf.nn.softmax))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def model_Train(X_train, Y_train, X_test, Y_test):
    model = build_model(X_train.shape[1:])
    model.fit(X_train, Y_train, epochs=200, batch_size=100, validation_data=(X_test, Y_test))

    model.save('../data/model/test_problemsModel.h5')

    return model

def model_evaluate(model, X_test, Y_test):
    score = model.evaluate(X_test, Y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])


if __name__ == "__main__":
    main()