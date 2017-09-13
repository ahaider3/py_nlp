import sys
sys.path.insert(0, '/home/cc/pytweet/')
import pickle
import json
import pandas as pd
from src import *


## use a pandas df to store (screen_name, location , tweet_id) -- aggregate this data as I stream

def get_target(target):
    tweets = api.user_timeline(target, target,target)
    rts_id = [ (tweet.id, api.retweets(tweet.id, 100)) for tweet in tweets]
    flat_rt_id = [(tup[0], rt) for tup in rts_id for rt in tup[1]]
    with_users = [ (rt[1].author.screen_name, api.get_user(rt[1].author.screen_name, 
                                                        rt[1].author.screen_name,
                                                        rt[1].author.screen_name).location,
                                        rt[0])
                                        for rt in flat_rt_id]
    res_frame = pd.DataFrame()
    res_frame["id"] = [tup[2] for tup in with_users]
    res_frame["name"] = [tup[0] for tup in with_users]
    res_frame["location"] = [tup[1] for tup in with_users]
    res_frame.to_csv(target + ".df", sep='\t')

    print( target, "Got:" ,len(with_users))
#    print(rts[0].author.screen_name)
    
#    screen_names = [ rt.author.screen_name for  rt in rts]
#    print(screen_names)
#    users = [ api.get_user(name, name, name) for name in screen_names]
#    locations = [ user.location for user in users]
    


def main(argv):
    tweet_list = []
    init = initialize.Initializer()
    init.initialize()
    stream = init.get_stream()
    global api
    api = init.api
    for i in argv:
        get_target(i)



if __name__ == "__main__":
    main(sys.argv[1:])
