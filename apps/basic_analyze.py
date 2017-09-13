import sys
sys.path.insert(0, '/home/cc/pytweet/')
import pickle
import json
import pandas as pd
from src import *
def get_most_followed(filtered_tweets):
    max_followers = filtered_tweets['Follower_count'].max()
    pos = filtered_tweets.ix[filtered_tweets['Follower_count'].idxmax()]['Name']
    return (pos, max_followers)
#    return max_pair

def pre_process(filtered_tweets):
    filtered_tweets.sort_values(['Follower_count'], ascending=False, inplace=True)
def get_top_k(filtered_tweets,k):
    
    top_names = filtered_tweets.head(k)['Name'].tolist()
    top_followers = filtered_tweets.head(k)['Follower_count'].tolist()
    top_text = filtered_tweets.head(k)['Text'].tolist()

    return [(top_names[i], top_followers[i], top_text[i]) for i in range(len(top_names))]

def add_sentiments(analyzer, df):
    texts = df['Text'].tolist()
    res = analyzer.classify(texts)
    df['Sentiment'] = res



def main(argv):
    tweet_list = []
    bs = basic_sentiment_sk.BasicSentimentSK(argv[1])
 #   bs.clean_words(3)
#    bs.get_word_features()
    bs.extract_features()
    print("Training")
    bs.train()
    bs.classify(["i do not hate the world"])
    print("Trained")
    
    columns=["Name", "Follower_count", "Text"]
    file_tweets = open(argv[0], "r")
    filtered_tweets = pd.DataFrame()
    curr_pos = 0
    num_failed = 0
    tweets=[]
    for line in file_tweets:
        res = 0
        try:
            tweet = json.loads(line)
            name = tweet['user']['name']
            follower_count = tweet['user']['followers_count']
            text = tweet['text']
            res = 1
        except:
            num_failed +=1

        if res:
            tweets.append((name, follower_count, text))
            curr_pos += 1
#            if name in filtered_tweets:
#                filtered_tweets[name]['tweets'].append(text)
#            else:
#                filtered_tweets[name] = {'tweets': [], 'followers_count': follower_count}
#                filtered_tweets[name]['tweets'].append(text)
#                filtered_tweets[name]['followers_count'] = follower_count
            
    print("Num Tweets:", len(tweets), "NumFailed:", num_failed)
    filtered_tweets['Name'] = [x[0] for x in tweets]
    filtered_tweets['Follower_count'] = [x[1] for x in tweets]
    filtered_tweets['Text'] = [x[2] for x in tweets]
    get_most_followed(filtered_tweets)
    pair = get_most_followed(filtered_tweets)
    print("Most Followed:", pair[0], "With:", pair[1])
    pre_process(filtered_tweets)
    tops = get_top_k(filtered_tweets, 20)
    for pair in tops:
        print(pair, "Has Sentiment:", bs.classify([pair[2]]))
    add_sentiments(bs, filtered_tweets)
    print("Overall Reaction from the World:", filtered_tweets['Sentiment'].mean())
    print("Correlation", filtered_tweets[['Follower_count', 'Sentiment']].corr())


if __name__ == "__main__":
    main(sys.argv[1:])
