import matplotlib.pyplot as plt
import deepLearningEngine.vgg16Network as his

acc = his.histroy.history['acc']
val_acc = his.histroy.history['val_acc']
loss = his.histroy.history['loss']
val_loss = his.histroy.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.legend()


plt.show()