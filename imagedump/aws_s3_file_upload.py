import boto3
import os

def s3FileUpload():
    imgFileList = os.listdir('./imgrename')[1:]

    bucket = 'wonriedu-aimapp-dev'
    for imgFile in imgFileList:
        uploadFile = os.path.join('./imgrename', imgFile)
        s3Path = 'problems/image/' + imgFile
        s3 = boto3.client('s3')
        s3.upload_file(uploadFile, bucket, s3Path, ExtraArgs={'ContentType':"image/png", 'ACL':"public-read"})