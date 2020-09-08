from PIL import Image
from pytesseract import *
from deepLearningEngine.bayesianEngine import BayesianFilter
from urllib.request import urlopen
import boto3
from server.mongoDBCon import *

bf = BayesianFilter()
# 전역 선언
global probImgeUrl
probImgeUrl = []

global probId
probId = []


# S3접속
def s3_Connection():
    s3Client = boto3.client('s3')
    s3resource = boto3.resource('s3', aws_access_key_id='AKIAIUFHMM3DQ6F2L3XQ', aws_secret_access_key='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')
    bucket = s3resource.Bucket('deeplearning-training-data-classification')

    for obj in bucket.objects.all():
        pathSplit = obj.key.split('/')
        if pathSplit[0] != '':
            urlPath = 'https://s3.ap-northeast-2.amazonaws.com/deeplearning-training-data-classification/' + obj.key
            probId.append(pathSplit[1])

            probImgeUrl.append(urlPath)


# OCR 추출 후 training
def ocrToTxt(fileId, fileName, lang='eng'):

    img = Image.open(urlopen(fileName))

    # 추출
    outText = image_to_string(img, lang=lang, config='--psm 6 -c preserve_interword_spaces=1')

    # training
    bf.fit(outText, fileId)

if __name__ == "__main__":

    books = dbCon('books', {})
    bookIds = []
    for book in books:
        bookIds.append(book.get('_id'))

    for bookId in bookIds:
        problems = dbCon('problems', {"content.picture": {"$regex": "https"}, "bookId": bookId})
        for problem in problems:
            probId.append(problem.get('_id'))
            probImgeUrl.append(problem.get('content').get('picture'))

            try:
                prefix = 'problem/' + str(problem.get('_id')) + '/'
                result = s3_Connection.s3Client.list_objects(Bucket='deeplearning-training-data-classification', Prefix=prefix,
                                               Delimiter='/')
                for i in result.get("Contents"):
                    print('sub folder : ', i.get('Key'))
            except:
                pass

    # OCR 추출 작업 메인
    for fullName in range(len(probImgeUrl)):
        # 한글+영어 추출(kor, eng , kor+eng)
        ocrToTxt(probId[fullName], probImgeUrl[fullName], 'kor+eng')

    # 작업 완료 메시지
    print('+++ Text Convert Complete! +++')
    bf.word_save()