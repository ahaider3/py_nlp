#!/bin/bash


cd /home/cc/aws/py_nlp/

export LD_LIBRARY_PATH=/home/cc/tf/old/tensorflow/third_party/mkl/:/opt/intel/mkl/lib/intel64/:/home/cc/cuda/lib64/:$LD_LIBRARY_PATH
export PATH=/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/:$PATH
export PATH=/opt/intel/intelpython2/bin/:$PATH

export PYTHONPATH=/home/cc/aws/py_nlp:$PYTHONPATH


python analysis/training/site_classification.py ~/sent_train/GoogleNews-vectors-negative300.bin
