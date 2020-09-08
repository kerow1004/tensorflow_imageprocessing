import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from urllib.request import urlopen
import boto3

from pymongo import MongoClient

probImgeUrl = []
categories = []
urlLists = []

# Mongodb 접속 접속
client = MongoClient('mongodb://mongomath:znqjspxltmMongo@m-dev-rs0-primary.thewonri.com:27017/math?authSource=admin')
db = client['math']
# 쿼리로 데이터 호출
collection = db.problems
problems = collection.find({"content.picture":{"$regex":"https"}}, {"content.picture":1})


# S3접속
s3Client = boto3.client('s3')
s3Resource = boto3.resource('s3', aws_access_key_id='AKIAIUFHMM3DQ6F2L3XQ',
                            aws_secret_access_key='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')
bucket = s3Resource.Bucket('deeplearning-training-data-classification')

# 분류 대상 카테고리
image_size_x = 64
image_size_y = 64
X = []
Y = []
# S3datas
for obj in bucket.objects.all():
    if obj.key.split('/')[0] != '' or obj.key.split('/')[2] != '':
        urlLists.append(obj.key)
for urlList in urlLists:
    probId = urlList.split('/')[1]
    imgPath = s3Client.get_object(Bucket='deeplearning-training-data-classification', Key=urlList)
    try:
        s3Img = Image.open(imgPath['Body'])
        s3Img = s3Img.convert("RGB")
        s3Img = s3Img.resize((image_size_x, image_size_y))
        data = np.asarray(s3Img)
        X.append(data)
        Y.append(probId)
    except:
        pass

# mongoDB data
for problem in problems:
    probImgeUrl.append(problem.get('content').get('picture'))
    urlImg = Image.open(urlopen(problem.get('content').get('picture')))
    urlImg = urlImg.convert("RGB")
    urlImg = urlImg.resize((image_size_x, image_size_y))
    data = np.asarray(urlImg)
    X.append(data)
    Y.append(problem.get('_id'))


X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = \
    train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./imageTrainingData/problem_makedata.npy", xy)
print("ok,", len(Y))