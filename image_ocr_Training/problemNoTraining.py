from PIL import Image, ImageOps
from pytesseract import *
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



# OCR 추출 후 training
def ocrToTxt(fileId, fileNo, fileName, fileType, lang='eng'):

    img = Image.open(urlopen(fileName))
    if fileType == '필수예제' or fileType == '발전예제' or fileType == '교과서 유형 흐름잡기':
        cropImg = img.crop((0, 0, 100, 50))
        grayImg = ImageOps.grayscale(cropImg)
        resultImg = ImageOps.invert(grayImg)

    else:
        cropImg = img.crop((0, 0, 50, 50))
        resultImg = ImageOps.grayscale(cropImg)

    # 추출
    outText = image_to_string(resultImg, lang=lang, config='-l eng --oem 1 --psm 6')

    resultImg.save(os.path.join('./numberImg/'+str(fileId)+'-'+fileNo+'.png'))

    print(fileNo, ':' ,outText)

    # training
    # bf.fit(outText, fileId)

if __name__ == "__main__":
    problems = dbCon('problems', {"content.picture":{"$regex":"https"}})
    for problem in problems:
        probId.append(problem.get('_id'))
        probNo.append(problem.get('problemNo'))
        probImgeUrl.append(problem.get('content').get('picture'))
        probType.append(problem.get('problemType'))

    # OCR 추출 작업 메인
    for fullName in range(len(probImgeUrl)):
        # 한글+영어 추출(kor, eng , kor+eng)
        ocrToTxt(probId[fullName], probNo[fullName], probImgeUrl[fullName], probType[fullName])

    # 작업 완료 메시지
    print('+++ Text Convert Complete! +++')
    # bf.word_save()