import os

from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

probNo = []
root_dir = "../numberImgtriandata/train"
for i in os.listdir(root_dir):
    probNo.append(i)
nb_classes = len(probNo)

def main():
    train_dir = '../numberImgtriandata/train'
    validation_dir = '../numberImgtriandata/validation'
    test_dir = '../numberImgtriandata/test'

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20,
                                                        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20,
                                                            class_mode='binary')
    model_Train(train_generator, validation_generator)

def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    # model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def model_Train(train_generator, validation_generator):
    model = build_model((150, 150, 3))
    # model.fit(X_train, Y_train, epochs=50, batch_size=50, validation_data=(X_test, Y_test))
    histroy = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                                  validation_data=validation_generator, validation_steps=50)
    model.save('../data/model/numberCropModel.h5')

    return model

def model_evaluate(model, X_test, Y_test):
    score = model.evaluate(X_test, Y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])


if __name__ == "__main__":
    main()