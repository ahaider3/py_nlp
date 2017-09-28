cd /home/cc/aws/py_nlp

export LD_LIBRARY_PATH=/home/cc/tf/old/tensorflow/third_party/mkl/:/opt/intel/mkl/lib/intel64/:/home/cc/cuda/lib64/:$LD_LIBRARY_PATH
export PATH=/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/:$PATH
export PATH=/opt/intel/intelpython2/bin/:$PATH


python apps/get_news.py cnn bloomberg al-jazeera-english the-huffington-post breitbart-news cnbc bbc-news the-economist the-wall-street-journal


