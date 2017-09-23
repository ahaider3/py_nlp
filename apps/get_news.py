import requests
#from bs4 import BeautifulSoup
import sys
import nltk
from nltk.collocations import *
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from rake_nltk import Rake
from mongo_funcs import write, delete
from nltk import bigrams

API_KEY="1ef6cd29ea0f4fd1a6d126b1d76495f7"


def normalize_str(text):
  stop = stopwords.words('english') + list(string.punctuation)
  return ' '.join([s for s in word_tokenize(text.lower()) if s not in stop])

  

def transform(s):
    while True:
        try:
            r = requests.get("https://newsapi.org/v1/articles?source=" + s +"&sortBy=top&apiKey="+
                API_KEY)
            res = r.json()
            return res
        except:
            return None



def news_getter(source):

    articles = [(s, transform(s)) for s in source]
    details = [ (source, news['title'], news['description'], news['urlToImage'])  for source, article in articles if "articles" in article  for news in article["articles"] ]
    return [ (source, title, desc, url) for source, title,desc,url in details if title is not None and desc is not None]


def find_common(descriptions):
    corpus = ""
    norm_desc = [(src, normalize_str(i), normalize_str(j), k, i, j) for src, i, j, k in descriptions]
    for src, i, j, k, l, m in norm_desc:
        corpus += " " + i + " \n " + j + " "
#    print(corpus)
    r = Rake()
    r.extract_keywords_from_text(corpus)
    keywords = r.get_ranked_phrases()
#    print(keywords)
    appends = []
    flag = 0
    print(keywords[:10])
    temp_k = [ ' '.join(big) for k in keywords[:10] for big in list(bigrams(k.split()))]
    print(temp_k)
    keywords = temp_k
#    print(corpus)
    for src, title, desc, url, old_title, old_desc in norm_desc:
      for word in keywords:
#        for word in keyword.split():
#        print((word, title))
        if word in title or word in desc:
          flag = 1
          appends.append((src, title, desc, url, old_title, old_desc))
          break
        if flag:
          flag = 0
          break
#    print(appends)
    print(len(appends), len(norm_desc))
#    print(norm_desc[1])
    data = [{"title":a[1], "description":a[2], "url_image":a[3], "source": a[0], "old_title": a[4], "old_description": a[5]} for a in appends]
    s = set()
    dedup_data = []
    for d in data:
      if d['old_title'] not in s:
        dedup_data.append(d)
        s.add(d['old_title'])
#    print(data)
    delete()
    write(dedup_data)
    write(dedup_data, "total_news")
    return appends
#    bigram_measures = nltk.collocations.BigramAssocMeasures()
#    nltk.download("punkt")
#    finder = BigramCollocationFinder.from_words(nltk.word_tokenize(corpus))
#    finder.apply_freq_filter(3) 
 #   final_res = finder.nbest(bigram_measures.pmi, 10)
#    return final_res

def main(argv):
    find_common(news_getter(argv))


if __name__ == "__main__":
    main(sys.argv[1:])


