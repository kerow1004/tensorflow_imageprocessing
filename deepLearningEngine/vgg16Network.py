from keras import layers, models, optimizers
from keras.applications import VGG16, ResNet50, DenseNet201
import testTrans


conv_base = DenseNet201(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

histroy = model.fit_generator(testTrans.train_generator, steps_per_epoch=100, epochs=30, validation_data=testTrans.validation_generator, validation_steps=50)
model.save('sample_model.h5')




