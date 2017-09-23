#!/bin/bash

cd /home/cc/pytweet

export LD_LIBRARY_PATH=/home/cc/tf/old/tensorflow/third_party/mkl/:/opt/intel/mkl/lib/intel64/:/home/cc/cuda/lib64/:$LD_LIBRARY_PATH
export PATH=/opt/intel/compilers_and_libraries_2017/linux/bin/intel64/:$PATH
export PATH=/opt/intel/intelpython2/bin/:$PATH

cd /home/cc/pytweet

python apps/get_news.py cnn bloomberg al-jazeera-english the-huffington-post breitbart-news cnbc bbc-news the-economist the-wall-street-journal


timeout -sHUP 10m python apps/tweet_gen.py >  ~/tweets/test.txt

#python apps/tweet_join.py /home/cc/tweets/test.txt ~/sent_train/GoogleNews-vectors-negative300.bin ~/temp_model/logreg-990000

python analysis/pipeline/user_classification_inference.py /home/cc/tweets/test.txt ~/sent_train/GoogleNews-vectors-negative300.bin rf_test_500 ~/temp_model/logreg-15360000


python apps/news_analysis.py ~/sent_train/GoogleNews-vectors-negative300.bin ~/temp_model/subj/logreg-5960000





