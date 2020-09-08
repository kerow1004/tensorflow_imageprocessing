from PIL import Image
from pytesseract import *
from deepLearningEngine.bayesianEngine import BayesianFilter
from urllib.request import urlopen
import boto3
from server.mongoDBCon import *

bf = BayesianFilter()

s3Client = boto3.client('s3')


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
        # bookId = 4
        probImgeUrl = []
        probId = []
        problems = dbCon({"content.picture": {"$regex": "https"}, "bookId": bookId})
        for problem in problems:
            probId.append(problem.get('_id'))
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
            for i in range(len(probId)):
                print(probId[i], ':', probImgeUrl[i])

        # OCR 추출 작업 메인
        for fullName in range(len(probImgeUrl)):
            # 한글+영어 추출(kor, eng , kor+eng)
            ocrToTxt(probId[fullName], probImgeUrl[fullName], 'kor+eng')
            print(probId[fullName], ':', probImgeUrl[fullName])

        bf.word_save(bookId)
        print('+++ Text Convert Complete! +++', bookId)
        # 작업 완료 메시지
