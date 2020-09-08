from PIL import Image, ImageOps
from pytesseract import *
from deepLearningEngine.bayesianEngine import BayesianFilter
from urllib.request import urlopen
import boto3
import os, datetime
from server.mongoDBCon import *

bf = BayesianFilter()

s3Client = boto3.client('s3')
def imgCrop(fileId, fileNo, fileName):
    now = datetime.datetime.now()
    strNow = now.strftime('%y%m%d%H%M%S%f')
    if fileName.split('/')[3] == 'deeplearning-training-data-classification':
        img = Image.open(urlopen(fileName))
        reSize = img.resize((int(img.width/5), int(img.height/5)))
        resultImg = reSize.crop((0, 0, 100, 50))
    else:
        img = Image.open(urlopen(fileName))
        resultImg = img.crop((0, 0, 100, 50))
    # grayImg = ImageOps.grayscale(cropImg)
    # resultImg = ImageOps.invert(grayImg)
    if os.path.isdir('../numberImg/'+ fileNo) == False:
        os.mkdir('../numberImg/'+ fileNo)
        resultImg.save(os.path.join('../numberImg/'+ fileNo + '/' +fileNo+'-'+strNow+'.png'))
    else:
        resultImg.save(os.path.join('../numberImg/'+ fileNo + '/' +fileNo+'-'+strNow+'.png'))



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

            # try:
            #     prefix = 'problem/' + str(problem.get('_id')) + '/'
            #     result = s3Client.list_objects(Bucket='deeplearning-training-data-classification', Prefix=prefix,
            #                                    Delimiter='/')
            #     for obj in result.get("Contents"):
            #         pathSplit = obj.get('Key').split('/')
            #         if pathSplit[2] != '':
            #             urlPath = 'https://s3.ap-northeast-2.amazonaws.com/deeplearning-training-data-classification/' + obj.get(
            #                 'Key')
            #             probId.append(pathSplit[1])
            #             probImgeUrl.append(urlPath)
            #             # print('sub folder : ', obj.get('Key'))
            # except:
            #     pass

        # OCR 추출 작업 메인
        for fullName in range(len(probImgeUrl)):
            try:
                if probNo[fullName].find('-') == 1:
                    imgCrop(probId[fullName], probNo[fullName].replace('-', '000000'), probImgeUrl[fullName])
                    print(probId[fullName], ':', probNo[fullName], ':', probImgeUrl[fullName])
                else:
                    imgCrop(probId[fullName], probNo[fullName], probImgeUrl[fullName])
                    print(probId[fullName], ':', probNo[fullName], ':', probImgeUrl[fullName])
            except:
                pass

        print('+++ Crop Complete! +++', problemNo)
        # 작업 완료 메시지
