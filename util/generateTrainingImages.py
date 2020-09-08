import PIL
import os
import math
from PIL import Image, ImageOps, ImageEnhance
from os.path import isdir, isfile, join
import numpy as np
import cv2 as cv
from datetime import datetime
from scipy.stats import truncnorm

imagePath = '../images/original'
# imageWidth = 224
# imageHeight = 224
outputPath = '../images/trainingSet'
# outputPath = './trainingSet'

# Black, White only palette(to quantize image)
palImage = Image.new('P', (1,1))
bwPal = [
  0,0,0,
  255,255,255,
] + [0,] * 254 * 3
palImage.putpalette(bwPal)

# https://stackoverflow.com/a/44308018
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

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

tempMargin = 30

def rotate(image, degree):
    tmpImage = image
    if degree < 0:
        tmpImage = Image.new('L', (image.width + tempMargin, image.height + tempMargin), '#ffffff')
        tmpImage.paste(image, (tempMargin, tempMargin))
    return image.rotate(degree, fillcolor='#ffffff')

def quad(image, height):
    return image.transform(image.size, PIL.Image.QUAD,
                           data=(0, 0, 0, image.height, image.width, image.height + height, image.width, -1 * height),
                           fillcolor='#ffffff')

def shearX(image, tan):
    tmpImage = image
    if tan < 0:
        tmpImage = Image.new('L', (image.width + tempMargin, image.height + tempMargin), '#ffffff')
        tmpImage.paste(image, (tempMargin, tempMargin))
    return tmpImage.transform(tmpImage.size, PIL.Image.AFFINE,
                           data=(1, tan, 0, 0, 1, 0),
                           fillcolor='#ffffff')

def shearY(image, tan):
    tmpImage = image
    if tan < 0:
        tmpImage = Image.new('L', (image.width + tempMargin, image.height + tempMargin), '#ffffff')
        tmpImage.paste(image, (tempMargin, tempMargin))
    return tmpImage.transform(tmpImage.size, PIL.Image.AFFINE,
                           data=(1, 0, 0, tan, 1, 0),
                           fillcolor='#ffffff')

def move(image, x, y):
    if x < 0 or y < 0:
        tmpImage = Image.new('L', (image.width + tempMargin, image.height + tempMargin), '#ffffff')
        tmpImage.paste(image, (tempMargin, tempMargin))
    return image.transform(image.size, PIL.Image.AFFINE,
                           data=(1, 0, x, 0, 1, y),
                           fillcolor='#ffffff')

def resize(image, x, y):
    return image.transform(image.size, PIL.Image.AFFINE,
                           data=(x, 0, 0, 0, y, 0),
                           fillcolor='#ffffff')

def crop(image, x, y):
    return image.crop((0, 0, image.width * x, image.height * y))

# perform horizontal warp like book
# height1 > 0
# 0.2 < checkpoint < 0.8
# height2 < 0
def warp(image, checkpoint, height1, height2):
    width, height = image.size
    bottomMargin = 0
    if abs(height1) < abs(height2):
        bottomMargin = abs(height2) - abs(height1)
    tempImage = Image.new('L', (width, height + height1 + bottomMargin), 255)

    checkWidth = round(width * checkpoint)

    for x in range(width):
        for y in range(height):
            if x < checkWidth:
                offset_y = round(height1 * math.sin(math.radians(90 * x / checkWidth)))
                # if y + offset_y < height:
                tempImage.putpixel((x, y + height1 - offset_y), image.getpixel((x, y)))
            else:
                offset_y = round(height2 * math.sin(math.radians(90 + 90 * (x - checkWidth) / (width - checkWidth))))
                # print(x, offset_y, y - height2 + offset_y)
                tempImage.putpixel((x, y - height2 + offset_y), image.getpixel((x, y)))

    return tempImage

def getRandomTransformedImage(image):
    checkpointNormalizer = get_truncated_normal(mean=0.5, sd=0.2, low=0.1, upp=0.9)
    warpCheckpoint = checkpointNormalizer.rvs()
    warpHeight1 = round(abs(np.random.normal(scale=0.1)) * image.height)
    warpHeight2 = round(-1 * abs(np.random.normal(scale=0.1)) * image.height)
    rotateDegree = np.random.normal(scale=4.0)
    quadHeight = np.random.normal(scale=5.0)
    isX = np.random.randint(2)
    affineTan = np.random.normal(scale=0.03)
    moveX = abs(np.random.normal(scale=5.0))
    moveY = abs(np.random.normal(scale=5.0))
    # isResize = np.random.rand()
    # resizeX = np.random.normal(loc=1.0, scale=0.02)
    # resizeY = np.random.normal(loc=1.0, scale=0.02)
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
    workImg = workImg.convert('L')

    workImg = warp(workImg, warpCheckpoint, warpHeight1, warpHeight2)

    workImg = move(workImg, moveX, moveY)
    workImg = quad(workImg, quadHeight)

    if isX == 1:
        workImg = shearX(workImg, affineTan)
    else:
        workImg = shearY(workImg, affineTan)

    workImg = rotate(workImg, rotateDegree)

    # this is meaningless
    # if isResize > .8:
    #     workImg = resize(workImg, resizeX, resizeY)

    if isCrop > .5:
        workImg = crop(workImg, cropX, cropY)

    #if workImg.width < 512 or workImg.height < 512:
    workImg = workImg.resize((400, 400), PIL.Image.LANCZOS)

    workImg = workImg.convert('RGB')
    return workImg

if isdir(outputPath) == False:
    os.mkdir(outputPath)

for index, id in enumerate(os.listdir(imagePath)):
    if id == '.DS_Store':
       continue
    if id.isnumeric():
        basePath = join(imagePath, id)
        baseFile = join(basePath, 'base.png')
        baseImg = Image.open(baseFile)

        baseImg = baseImg.resize((baseImg.width * 2, baseImg.height * 2), PIL.Image.LANCZOS)

        fileIndex = 0

        for x in range(500):
            fileIndex += 1
            if isfile(join(outputPath, id, str(fileIndex) + '.jpg')):
                continue
            workImg = getRandomTransformedImage(baseImg)

            if isdir(join(outputPath, id)) == False:
                os.mkdir(join(outputPath, id))

            workImg.save(join(outputPath, id, str(fileIndex) + '.jpg'))

        print(datetime.now(), id)
'''
        pictures = [f for f in os.listdir(basePath) if isfile(join(basePath, f)) and f.startswith('picture')]
        for index, picture in enumerate(pictures):
            pictureFile = join(basePath, picture)
            pictureImg = Image.open(pictureFile)

            resizedWidth = 800
            resizedHeight = math.floor(pictureImg.height * resizedWidth / pictureImg.width)
            # for fast manipulation
            resizedImage = pictureImg.resize((resizedWidth, resizedHeight), PIL.Image.LANCZOS)

            for x in range(400):
                workImg = getRandomTransformedImage(resizedImage)

                fileIndex += 1

                workImg.save(join(outputPath, id, str(fileIndex) + '.jpg'))

'''


