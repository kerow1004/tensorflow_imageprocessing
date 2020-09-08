from pymongo import MongoClient

def dbCon(col, query):
    # Mongodb 접속
    client = MongoClient('mongodb://mongomath:znqjspxltmMongo@m-dev-rs0-primary.thewonri.com:27017/math?authSource=admin')
    db = client['math']
    result = db.get_collection(col).find(query)

    return result
