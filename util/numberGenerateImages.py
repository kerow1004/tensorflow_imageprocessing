import PIL
import os
import math
from PIL import Image, ImageOps, ImageEnhance
from os.path import isdir, isfile, join
import numpy as np
import cv2 as cv
import datetime

imagePath = '/Volumes/G-DRIVE mobile/numberImg'
# imageWidth = 22
# imageHeight = 224
outputPath = '/Volumes/G-DRIVE mobile/numberImg'

# Black, White only palette(to quantize image)
palImage = Image.new('P', (1,1))
bwPal = [
  0,0,0,
  255,255,255,
] + [0,] * 254 * 3
palImage.putpalette(bwPal)

# https://stackoverflow.com/questions/29433243/convert-image-to-specific-palette-using-pil-without-dithering
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

def rotate(image, degree):
    return image.rotate(degree, fillcolor='#ffffff')

def quad(image, height):
    return image.transform(image.size, PIL.Image.QUAD,
                           data=(0, 0, 0, image.height, image.width, image.height + height, image.width, -1 * height),
                           fillcolor='#ffffff')

def shearX(image, tan):
    return image.transform(image.size, PIL.Image.AFFINE,
                           data=(1, tan, 0, 0, 1, 0),
                           fillcolor='#ffffff')

def shearY(image, tan):
    return image.transform(image.size, PIL.Image.AFFINE,
                           data=(1, 0, 0, tan, 1, 0),
                           fillcolor='#ffffff')

def move(image, x, y):
    return image.transform(image.size, PIL.Image.AFFINE,
                           data=(1, 0, x, 0, 1, y),
                           fillcolor='#ffffff')

def resize(image, x, y):
    return image.transform(image.size, PIL.Image.AFFINE,
                           data=(x, 0, 0, 0, y, 0),
                           fillcolor='#ffffff')

def crop(image, x, y):
    return image.crop((0, 0, image.width * x, image.height * y))

def getRandomTransformedImage(image):
    rotateDegree = np.random.normal(scale=2.0)
    quadHeight = np.random.normal(scale=5.0)
    isX = np.random.randint(2)
    affineTan = np.random.normal(scale=0.03)
    moveX = np.random.normal(scale=5.0)
    moveY = np.random.normal(scale=5.0)
    isResize = np.random.rand()
    resizeX = np.random.normal(loc=1.0, scale=0.02)
    resizeY = np.random.normal(loc=1.0, scale=0.02)
    isCrop = np.random.rand()
    cropX = 1.0 - abs(np.random.normal(scale=0.2))
    cropY = 1.0 - abs(np.random.normal(scale=0.2))

    workImg = image.copy()

    # Binarize image
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    cvImg = np.asarray(workImg.convert('L'))
    # bwImg = cv.adaptiveThreshold(cvImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    ret, bwImg = cv.threshold(cvImg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    workImg = Image.fromarray(bwImg)
    workImg = move(workImg, moveX, moveY)
    workImg = quad(workImg, quadHeight)

    if isX == 1:
        workImg = shearX(workImg, affineTan)
    else:
        workImg = shearY(workImg, affineTan)

    workImg = rotate(workImg, rotateDegree)

    if isResize > .8:
        workImg = resize(workImg, resizeX, resizeY)

    if isCrop > .8:
        workImg = crop(workImg, cropX, cropY)

    return workImg

if isdir(outputPath) == False:
    os.mkdir(outputPath)

imgList = []
for i in os.listdir('/Volumes/G-DRIVE mobile/numberImg'):
    if i == '.DS_Store' or i=='aaa':
        continue
    if len(os.listdir(os.path.join('/Volumes/G-DRIVE mobile/numberImg', i))) <= 9000:
        imgList.append(i)

for index, id in enumerate(imgList):
    if id == '.DS_Store':
       continue
    try:
        for img in os.listdir('/Volumes/G-DRIVE mobile/numberImg/'+id):
            basePath = join(imagePath, id)
            baseFile = join(basePath, img)
            baseImg = Image.open(baseFile)



            for x in range(0, 2):
                workImg = getRandomTransformedImage(baseImg)

                if isdir(join(outputPath, id)) == False:
                    os.mkdir(join(outputPath, id))

                now = datetime.datetime.now()
                strNow = now.strftime('%y%m%d%H%M%S%f')

                workImg.save(join(outputPath, id, strNow+ '.jpg'))
                print(join(outputPath, id, strNow+ '.jpg'))

            pictures = [f for f in os.listdir(basePath) if isfile(join(basePath, f)) and f.startswith('picture')]
            for index, picture in enumerate(pictures):
                pictureFile = join(basePath, picture)
                pictureImg = Image.open(pictureFile)

                resizedWidth = 800
                resizedHeight = math.floor(pictureImg.height * resizedWidth / pictureImg.width)
                # for fast manipulation
                resizedImage = pictureImg.resize((resizedWidth, resizedHeight), PIL.Image.LANCZOS)

                # TODO https://github.com/mzucker/page_dewarp
                # TODO find better method to quantize 2 color
                # qImage = quantizetopalette(resizedImage, palImage)
                # qImage = qImage.convert('RGB')
                # qImage.save(join(outputPath, id, str(0) + '.jpg'))

                for x in range(0, 2):
                    workImg = getRandomTransformedImage(resizedImage)

                    now1 = datetime.datetime.now()
                    strNow1 = now1.strftime('%y%m%d%H%M%S%f')
                    workImg.save(join(outputPath, id, strNow1 + '.jpg'))
    except :
        pass












