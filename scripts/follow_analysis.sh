#!/bin/bash


cd /home/cc/aws/py_nlp/

export LD_LIBRARY_PATH=/home/cc/tf/old/tensorflow/third_party/mkl/:/opt/intel/mkl/lib/intel64/:/home/cc/cuda/lib64/:$LD_LIBRARY_PATH
export PATH=/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/:$PATH
export PATH=/opt/intel/intelpython2/bin/:$PATH

export PYTHONPATH=/home/cc/aws/py_nlp:$PYTHONPATH


python apps/user_gen.py 10 10 100  ~/sent_train/GoogleNews-vectors-negative300.bin ~/aws/models/final/news_decipher-2180000 100 cnn FoxNews BreitbartNews
