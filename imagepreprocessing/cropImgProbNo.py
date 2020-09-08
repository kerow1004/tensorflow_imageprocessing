from PIL import Image, ImageOps
from deepLearningEngine.bayesianEngine import BayesianFilter
from urllib.request import urlopen
from server.mongoDBCon import *
import os


bf = BayesianFilter()
# 전역 선언
global probImgeUrl
probImgeUrl = []
global probNo
probNo = []
global probId
probId = []
global probType
probType = []



def imgCrop(fileId, fileNo, fileName):

    img = Image.open(urlopen(fileName))
    cropImg = img.crop((0, 0, 200, 150))
    grayImg = ImageOps.grayscale(cropImg)
    resultImg = ImageOps.invert(grayImg)
    # 추출
    # outText = image_to_string(resultImg, lang=lang, config='-l eng --oem 1 --psm 6')
    if os.path.isdir('./numberImg/'+ fileNo) == False:
        os.mkdir('./numberImg/'+ fileNo)
        resultImg.save(os.path.join('./numberImg/'+ fileNo + '/' +str(fileId)+'-'+fileNo+'.png'))
    else:
        resultImg.save(os.path.join('./numberImg/'+ fileNo + '/' +str(fileId)+'-'+fileNo+'.png'))

    # print(fileNo, ':' ,outText)

    # training
    # bf.fit(outText, fileId)

if __name__ == "__main__":
    problems = dbCon('problems', {"content.picture":{"$regex":"https"}})
    for problem in problems:
        probId.append(problem.get('_id'))
        probNo.append(problem.get('problemNo'))
        probImgeUrl.append(problem.get('content').get('picture'))

    for fullName in range(len(probImgeUrl)):
        imgCrop(probId[fullName], probNo[fullName], probImgeUrl[fullName])

    # 작업 완료 메시지
    print('+++ Text Convert Complete! +++')
    # bf.word_save()