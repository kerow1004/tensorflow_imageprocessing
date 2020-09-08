import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os, datetime, shutil

numberList = os.listdir('../numberImg/')
# numberList.remove('.DS_Store')

# 분류 대상 카테고리
image_size_x = 64
image_size_y = 64
X = []
Y = []

for idx, obj in enumerate(numberList):
    if obj != '.DS_Store':
        imageList = os.listdir('../numberImg/' + obj)
        # imageList.remove('.DS_Store')
        for no in imageList:
            img = Image.open('../numberImg/' + obj + '/' + no)
            img = img.convert("RGB")
            img = img.resize((image_size_x, image_size_y))
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
            for ang in range(-10, 10, 1):
                img2 = img.rotate(ang)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

now = datetime.datetime.now()
nowDataTime = now.strftime('%y%m%d%H%M')
shutil.move('../imageTraining/problem_makedata.npy', '../imageTraining/history/'+ nowDataTime +'.npy')

np.save("../imageTraining/problem_makedata.npy", xy)
print("ok,", len(Y))