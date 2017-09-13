from pymongo import MongoClient

client = MongoClient()
def write(data, coll_name="topnews"):
  db = client.test
  coll = db[coll_name]
  coll.insert_many(data)

def delete(coll_name="topnews"):
  db = client.test
  coll = db[coll_name]
  coll.remove({})

def read():
  db = client.test
  return db.topnews.find({})
