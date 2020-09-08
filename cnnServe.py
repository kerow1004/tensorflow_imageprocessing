from keras.models import load_model
import PIL
from PIL import Image
import os
from os.path import join, isfile
import pickle
import numpy as np

load_dir = os.getcwd()
model_name = 'cnn_basic_model.h5'
label_name = 'cnn_basic_label.pickle'

model = load_model(join(load_dir, model_name))
labels = {}
with open(join(load_dir, label_name), 'rb') as handle:
    labels = pickle.load(handle)

palImage = Image.new('P', (1,1))
bwPal = [
  0,0,0,
  255,255,255,
] + [0,] * 254 * 3
palImage.putpalette(bwPal)

def quantizetopalette(silf, palette, dither=False):
    """Convert an RGB or L mode image to use a given P image's palette."""

    silf.load()

    # use palette from reference image
    palette.load()
    if palette.mode != "P":
        raise ValueError("bad mode for palette image")
    if silf.mode != "RGB" and silf.mode != "L":
        raise ValueError(
            "only RGB or L mode images can be quantized to a palette"
            )
    im = silf.im.convert("P", 1 if dither else 0, palette.im)
    # the 0 above means turn OFF dithering

    return silf._new(im)

testImg = Image.open(join(load_dir, 'bw.jpg'))
#testImg = quantizetopalette(testImg, palImage)
testImg = testImg.resize((224, 224), PIL.Image.LANCZOS)
#rImg = testImg.convert('RGB')
#rImg.save(join(load_dir, 'bw.jpg'))
testImg = testImg.convert('L')
testData = np.asarray(testImg)
testData = testData.reshape(1, testImg.width, testImg.height, 1)

result = model.predict(testData, verbose=1)
print(result)

resultIndex = result.argmax()
print(resultIndex)

resultLabel = labels[resultIndex]
print(resultLabel)
