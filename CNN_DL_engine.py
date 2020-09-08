import os

from PIL import Image
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.applications import VGG16, ResNet50
from keras.utils import np_utils
from keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np
# from deepLearningEngine.CNN_model import build_model

probNo = []
root_dir = "./trainingSet"
for i in os.listdir(root_dir):
    probNo.append(i)
nb_classes = len(probNo)

image_w = 64
image_h = 64

def main():
    X_train, X_test, Y_train, Y_test = np.load("./test_makedata.npy")
    # 데이터 정규화하기
    X_train = X_train.astype("float32") / 256
    X_test  = X_test.astype("float32")  / 256
    #
    # Y_train = np_utils.to_categorical(y_train)
    # Y_test = np_utils.to_categorical(y_test)

    model = model_Train(X_train, Y_train, X_test, Y_test)
    model_evaluate(model, X_test, Y_test)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', Y_test.shape)

def build_model(input_shape):
    model = Sequential()
    model.add(VGG16(
        include_top=False,
                        weights='imagenet'
                       ,input_shape=input_shape
                       ))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.1))
    #
    # model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.3))
    #
    # model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation(tf.nn.softmax))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def model_Train(X_train, Y_train, X_test, Y_test):
    model = build_model(X_train.shape[1:])
    model.fit(X_train, Y_train, epochs=50, batch_size=50, validation_data=(X_test, Y_test))

    model.save('./data/model/numberCropModel.h5')

    return model

def model_evaluate(model, X_test, Y_test):
    score = model.evaluate(X_test, Y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])


if __name__ == "__main__":
    main()