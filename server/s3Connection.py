import boto3

def s3_Connection():
    s3Client = boto3.client('s3')
    s3resource = boto3.resource('s3', aws_access_key_id='AKIAIUFHMM3DQ6F2L3XQ', aws_secret_access_key='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')
    bucket = s3resource.Bucket('deeplearning-training-data-classification')