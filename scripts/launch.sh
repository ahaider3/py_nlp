#!/bin/bash


cd /home/cc/pytweet

python apps/get_news.py cnn bloomberg cnn al-jazeera-english abc-news-au breitbart-news


timeout -sHUP 30m python apps/tweet_gen.py >  ~/tweets/test.txt

python apps/tweet_join.py /home/cc/tweets/test.txt ~/sent_train/GoogleNews-vectors-negative300.bin ~/temp_model/logreg-990000

python apps/news_analysis.py ~/sent_train/GoogleNews-vectors-negative300.bin ~/temp_model/subj/logreg-5960000





