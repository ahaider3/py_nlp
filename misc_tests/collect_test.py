# test for collect operation in spark
from pyspark import SparkContext, SparkConf
from pyspark.serializers import MarshalSerializer
import numpy as np
import time
import sys
sys.path.insert(0, '/home/cc/pygraph/')
sys.path.insert(0, '/home/cc/spark-2.1.1-bin-hadoop2.7/python/')
sys.path.insert(0, '/home/cc/spark-2.1.1-bin-hadoop2.7/python/lib')
from functools import partial
from pyspark import *
from src import *



class Test:

  def __init__(self, appName, master,size):
    self.conf = SparkConf().setAppName(appName).setMaster(master)
    self.sc = SparkContext(conf=self.conf, serializer=MarshalSerializer())
    self.sc.setLogLevel("ERROR")
    self.msg_rdd = self.sc.parallelize(range(0,size))
    self.msg_rdd.cache()



def main(argv):
  comm = Test("test", "local", int(argv[0]))
  # cache it
  set_check = set(range(0,int(int(argv[0])/2)))
  set_check_b = comm.sc.broadcast(set_check)
  comm.msg_rdd.count()
 # comm.msg_rdd = comm.msg_rdd.filter(lambda k: k not in set_check_b.value)
 # comm.msg_rdd = comm.msg_rdd.map(lambda k: (k,k)).filter(lambda k: k[0] %2 == 0)
  comm.msg_rdd = comm.msg_rdd.map(lambda k: (k,k))


  start = time.time()
  comm.msg_rdd.count()

#  comm.msg_rdd.collect()
  end = time.time()
  print("Time for:", int(argv[0]), "IS", end-start)


if __name__ == "__main__":
  main(sys.argv[1:])



