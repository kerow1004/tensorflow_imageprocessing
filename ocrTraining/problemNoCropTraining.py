from PIL import Image, ImageOps
from pytesseract import *
from deepLearningEngine.bayesianEngine import BayesianFilter
from urllib.request import urlopen
import boto3
import os
from server.mongoDBCon import *

bf = BayesianFilter()

s3Client = boto3.client('s3')
def imgCrop(fileId, fileNo, fileName):

    img = Image.open(urlopen(fileName))
    cropImg = img.crop((0, 0, 200, 150))
    grayImg = ImageOps.grayscale(cropImg)
    resultImg = ImageOps.invert(grayImg)
    if os.path.isdir('./numberImg/'+ fileNo) == False:
        os.mkdir('./numberImg/'+ fileNo)
        resultImg.save(os.path.join('./numberImg/'+ fileNo + '/' +str(fileId)+'-'+fileNo+'.png'))
    else:
        resultImg.save(os.path.join('./numberImg/'+ fileNo + '/' +str(fileId)+'-'+fileNo+'.png'))

# OCR 추출 후 training
def ocrToTxt(fileId, fileName, lang='eng'):
    img = Image.open(urlopen(fileName))

    # 추출
    outText = image_to_string(img, lang=lang, config='--psm 6 -c preserve_interword_spaces=1')

    # training
    bf.fit(outText, fileId)


if __name__ == "__main__":
    collection = 'problems'
    problems = dbCon(collection, {"content.picture": {"$regex": "https"}})
    problemNos = set()
    for problem in problems:
        probNoValue = problem.get('problemNo')
        if probNoValue != '':
            problemNos.add(probNoValue)

    for problemNo in problemNos:
        # bookId = 4
        probImgeUrl = []
        probId = []
        probNo = []
        problems = dbCon(collection, {"content.picture": {"$regex": "https"}, "problemNo": problemNo})
        for problem in problems:
            probId.append(problem.get('_id'))
            probNo.append(problem.get('problemNo'))
            probImgeUrl.append(problem.get('content').get('picture'))

            try:
                prefix = 'problem/' + str(problem.get('_id')) + '/'
                result = s3Client.list_objects(Bucket='deeplearning-training-data-classification', Prefix=prefix,
                                               Delimiter='/')
                for obj in result.get("Contents"):
                    pathSplit = obj.get('Key').split('/')
                    if pathSplit[2] != '':
                        urlPath = 'https://s3.ap-northeast-2.amazonaws.com/deeplearning-training-data-classification/' + obj.get(
                            'Key')
                        probId.append(pathSplit[1])
                        probImgeUrl.append(urlPath)
                        # print('sub folder : ', obj.get('Key'))
            except:
                pass
            # for i in range(len(probId)):
            #     print(probId[i], ':', probImgeUrl[i])

        # OCR 추출 작업 메인
        for fullName in range(len(probImgeUrl)):
            # 한글+영어 추출(kor, eng , kor+eng)
            ocrToTxt(probId[fullName], probImgeUrl[fullName], 'kor+eng')
            imgCrop(probId[fullName], probNo[fullName], probImgeUrl[fullName])
            print(probId[fullName], ':', probImgeUrl[fullName])

        bf.word_save(problemNo)
        print('+++ Text Training Complete! +++', problemNo)
        # 작업 완료 메시지
