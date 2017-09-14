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

def read(coll="topnews"):
  db = client.test
  return db[coll].find({})
